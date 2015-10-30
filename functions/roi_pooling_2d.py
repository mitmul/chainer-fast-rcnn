import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ROIPooling2D(function.Function):

    """RoI pooling over a set of 2d planes."""

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_height, self.pooled_width = pooled_height, pooled_width
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.int32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_gpu(self, inputs):
        bottom_data, bottom_rois = inputs
        num, channels, height, width = bottom_data.shape
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.pooled_height,
                                    self.pooled_width), dtype=numpy.float32)
        self.argmax_data = cuda.cupy.empty_like(top_data).astype(numpy.int32)
        count = numpy.prod(top_data.shape)
        cuda.cupy.ElementwiseKernel(
            '''
            raw T rois, raw T pooled_height, raw T pooled_width,
            raw S spatial_scale, raw S bottom_data, raw T channels,
            raw T height, raw T width
            ''',
            'raw S top_data, raw T argmax_data',
            '''
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = rois[num * 5 + 0];
            int roi_start_w = round(rois[num * 5 + 1] * spatial_scale);
            int roi_start_h = round(rois[num * 5 + 2] * spatial_scale);
            int roi_end_w = round(rois[num * 5 + 3] * spatial_scale);
            int roi_end_h = round(rois[num * 5 + 4] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);
            S bin_size_h = static_cast<S>(roi_height)
                           / static_cast<S>(pooled_height);
            S bin_size_w = static_cast<S>(roi_width)
                           / static_cast<S>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<S>(ph))
                                          * bin_size_h);
            int wstart = static_cast<int>(floor(static_cast<S>(pw))
                                          * bin_size_w);
            int hend = static_cast<int>(floor(static_cast<S>(ph + 1))
                                        * bin_size_h);
            int wend = static_cast<int>(floor(static_cast<S>(pw + 1))
                                        * bin_size_w);

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            S maxval = is_empty ? 0 : -3.402823466E+38F;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int bottom_offset = (roi_batch_ind * channels + c) * height
                                * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = bottom_offset + h * width + w;
                    if (bottom_data[bottom_index] > maxval) {
                        maxval = bottom_data[bottom_index];
                        maxidx = bottom_index;
                    }
                }
            }
            top_data[i] = maxval;
            argmax_data[i] = maxidx;
            ''',
            'roi_poolig_2d_fwd'
        )(bottom_rois, self.pooled_height, self.pooled_width,
          self.spatial_scale, bottom_data, channels, height, width, top_data,
          self.argmax_data, size=count)

        return top_data,

    def backward(self, inputs, gy):
        pass


def roi_pooling_2d(x, rois, pooled_height=7, pooled_width=7,
                   spatial_scale=0.0625):
    return ROIPooling2D(pooled_height, pooled_width, spatial_scale)(x, rois)
