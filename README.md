# chainer-fast-rcnn

## Requirements

- [OpenCV 3.0 with python bindings](http://opencv.org)
- [Chainer 1.4](https://github.com/pfn/chainer)
- [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- [dlib](https://github.com/davisking/dlib)

## Create symlinks

Create a symlink from the location of original [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) dir to this project's root dir. (The below line assumes a environment variable `$FAST_RCNN_HOME` has a path to the `fast-rcnn` source dir.)

```
$ ln -s $FAST_RCNN_HOME ./
```

Make sure that some cython extentions have been built. If not,

```
$ cd $FAST_RCNN_HOME/lib
$ make
```

Create a symlink from [dlib](https://github.com/davisking/dlib) to this project's root dir. (The below line assumes a envirionment variable `$DLIB_HOME` contains a path to the `dlib` source dir.)

```
$ ln -s $DLIB_HOME ./
```

Make sure that `dlib` has been already built. If not,

```
$ cd $DLIB_HOME/python_examples
$ ./compile_dlib_python_module.bat
```

## Load net

```
$ python scripts/load_net.py
```

## Test

First you should prepare a sample image, and then

```
$ python scripts/forward.py --img_fn sample.jpg --out_fn result.jpg
```
