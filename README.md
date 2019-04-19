# NPU_CaffeSSD
RK3399 Pro NPU support for Caffe SSD detector


Assume you have installed RKNN convert tool or you can install it first by 

```
$ pip install rknn_toolkit-0.9.9-cp35-cp35m-linux_x86_64.whl
```
the package you can find at 
1. [[Main Site]](http://t.rock-chips.com/forum.php?mod=forumdisplay)
2. [[Download Source Site]](http://t.rock-chips.com/forum.php?mod=viewthread&tid=79&extra=page%3D1)

## Usage
extract priorbox 
  
```
$ python priorbox.py
```

convert model and run pedestrian detect(may take some time to run)
  
```
$ python3 npu_ssd_det.py
```

## Result

<p align="center">
    <img src="test.png" width="500"\>
</p>
