# chainer-fast-rcnn

## Requirements

- [OpenCV 3.0 with python bindings](http://opencv.org)
- [Chainer 1.4](https://github.com/pfn/chainer)
- [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- [dlib v18.18](https://github.com/davisking/dlib)

## Create symlink

Create a symlink from the location of original [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) dir to this project's root dir. (The below line assumes a environment variable `$FRCN_ROOT` has a path to the `fast-rcnn` source dir.)

```
$ ln -s $FRCN_ROOT ./
```

Make sure that all steps written in the `Installation (sufficient for the demo)` section of `README.md` in [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) have been performed.

## Convert model

Convert caffemodel to chainermodel.

```
$ python scripts/load_net.py
```

## Test

First you should prepare a sample image, and then

```
$ python scripts/forward.py --img_fn sample.jpg --out_fn result.jpg
```
