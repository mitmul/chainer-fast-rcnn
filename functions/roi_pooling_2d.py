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
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_gpu(self, inputs):
        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.pooled_height,
                                    self.pooled_width), dtype=numpy.float32)
        self.argmax_data = cuda.cupy.empty_like(top_data).astype(numpy.int32)
        count = numpy.prod(top_data.shape)
        cuda.cupy.ElementwiseKernel(
            '''
            raw S bottom_data, raw S spatial_scale, raw T channels,
            raw T height, raw T width, raw T pooled_height, raw T pooled_width,
            raw S bottom_rois
            ''',
            'raw S top_data, raw T argmax_data',
            '''
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_rois[num * 5 + 0];
            int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);
            int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);
            int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);
            int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);

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
            ''', 'roi_poolig_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.pooled_height, self.pooled_width, bottom_rois, top_data,
          self.argmax_data, size=count)

        return top_data,

    def backward_gpu(self, inputs, gy):
        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        count = numpy.prod(bottom_data.shape)
        bottom_diff = cuda.cupy.zeros_like(bottom_data)
        cuda.cupy.ElementwiseKernel(
            '''
            raw S top_diff, raw T argmax_data, raw T num_rois,
            raw S spatial_scale, raw T channels, raw T height, raw T width,
            raw T pooled_height, raw T pooled_width
            ''',
            'raw S bottom_diff, raw S bottom_rois',
            '''
            int w = i % width;
            int h = (i / width) % height;
            int c = (i / width / height) % channels;
            int num = i / width / height / channels;

            S gradient = 0;
            // Accumulate gradient over all ROIs that pooled this element
            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
                int roi_batch_ind = bottom_rois[roi_n * 5];
                // Skip if ROI's batch index doesn't match n
                if (num != roi_batch_ind) {
                    continue;
                }

                int roi_start_w = round(bottom_rois[roi_n * 5 + 1]
                                        * spatial_scale);
                int roi_start_h = round(bottom_rois[roi_n * 5 + 2]
                                        * spatial_scale);
                int roi_end_w = round(bottom_rois[roi_n * 5 + 3]
                                      * spatial_scale);
                int roi_end_h = round(bottom_rois[roi_n * 5 + 4]
                                      * spatial_scale);

                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                     h >= roi_start_h && h <= roi_end_h);
                if (!in_roi) {
                    continue;
                }

                int offset = (roi_n * channels + c) * pooled_height
                             * pooled_width;

                // Compute feasible set of pooled units that could have pooled
                // this bottom unit

                // Force malformed ROIs to be 1x1
                int roi_width = max(roi_end_w - roi_start_w + 1, 1);
                int roi_height = max(roi_end_h - roi_start_h + 1, 1);

                S bin_size_h = static_cast<S>(roi_height)
                               / static_cast<S>(pooled_height);
                S bin_size_w = static_cast<S>(roi_width)
                               / static_cast<S>(pooled_width);

                int phstart = floor(static_cast<S>(h - roi_start_h)
                                    / bin_size_h);
                int phend = ceil(static_cast<S>(h - roi_start_h + 1)
                                 / bin_size_h);
                int pwstart = floor(static_cast<S>(w - roi_start_w)
                                    / bin_size_w);
                int pwend = ceil(static_cast<S>(w - roi_start_w + 1)
                                 / bin_size_w);

                phstart = min(max(phstart, 0), pooled_height);
                phend = min(max(phend, 0), pooled_height);
                pwstart = min(max(pwstart, 0), pooled_width);
                pwend = min(max(pwend, 0), pooled_width);

                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int index = ph * pooled_width + pw + offset;
                        T argmax_pos = argmax_data[index];
                        if (argmax_pos == (h * width + w)) {
                            gradient += top_diff[index];
                        }
                    }
                }
                bottom_diff[i] = gradient;
            }
            ''', 'roi_pooling_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois.shape[0], self.spatial_scale,
          channels, height, width, self.pooled_height, self.pooled_width,
          bottom_diff, bottom_rois, size=count)

        return bottom_diff, None


def roi_pooling_2d(x, rois, pooled_height=7, pooled_width=7,
                   spatial_scale=0.0625):
    return ROIPooling2D(pooled_height, pooled_width, spatial_scale)(x, rois)
