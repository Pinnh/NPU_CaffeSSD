# NPU_CaffeSSD

RK3399 Pro NPU support for Caffe SSD detector

assume you have installed RKNN toolkit and it's dependencies or you can install it first by 

```
$ pip install rknn_toolkit-0.9.9-cp35-cp35m-linux_x86_64.whl
```
the package you can find at 
1. [[Main Site]](http://t.rock-chips.com/forum.php?mod=forumdisplay)
2. [[Download Source Site]](http://t.rock-chips.com/forum.php?mod=viewthread&tid=79&extra=page%3D1)

you can find the pedestrian detecttion model of this project at [[here]](https://github.com/zlingkang/mobilenet_ssd_pedestrian_detection).

## Usage

extract priorbox (you need change the **caffe_root** path to yours in **priorbox.py**, or may need install **caffe-ssd** first, caffe-ssd you can find at [[here]](https://github.com/weiliu89/caffe))
  
```
$ python priorbox.py
```

convert model and run pedestrian detection (may take some time to run)
  
```
$ python3 npu_ssd_det.py
```

## Result

you can use NMS algorithm to remove redundant rects.

<p align="center">
    <img src="test.png" width="600"\>
</p>

## Issues
1. In order to get priorbox ,when you change the detection model, you should modify model's prototxt looks like **MobileNetSSD_deploy_truncated.prototxt**

2. Different model has different **outlen** (npu_ssd_det.py),If you change the detection model, you should replace the **outlen** first,How to do this see [[issue]](https://github.com/Pinnh/NPU_CaffeSSD/issues/1)
