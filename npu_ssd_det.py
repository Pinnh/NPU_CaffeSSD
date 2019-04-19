from rknn.api import RKNN
import cv2
import numpy as np

import re
import math
import random
import matplotlib.pylab as plt


def caffe2rknn(caffe_proto,caffe_weight,rknn_model):
    print("start export")
    rknn=RKNN(verbose=True)
    ret=rknn.load_caffe(model=caffe_proto,
                    proto="caffe",
                    blobs=caffe_weight)

    rknn.config(channel_mean_value='127.5 127.5 127.5 128.0',
		reorder_channel='2 1 0',
                #reorder_channel='0 1 2',
                #need_horizontal_merge=True
                )
    ret = rknn.build(do_quantization=False)
    #ret = rknn.build(do_quantization=True)
    ret=rknn.export_rknn(export_path=rknn_model)
    print("export finished")


def run_ssd(img_path,priorbox_path):
    #caffe_proto="./MobileNetSSD_deploy.prototxt"
    caffe_proto= "./MobileNetSSD_deploy_truncated.prototxt"
    caffe_weight="./MobileNetSSD_deploy10695.caffemodel"

    rknn_model="./pedestrian_ssd.rknn"

    caffe2rknn(caffe_proto,caffe_weight,rknn_model)
    
    print("run ssd")
    rknn=RKNN(verbose=True)
    ret=rknn.load_rknn(path=rknn_model)
    ret=rknn.init_runtime()
    #ret = rknn.init_runtime(target='rk1808', device_id='012345789AB')

    img=cv2.imread(img_path)
    img=cv2.resize(img,(300,300))
    print("shape:",img.shape)
    outlen=7668 #change to your model

    priorbox=[]
    with open(priorbox_path) as f:
         for line in  f:
             arr=line.strip().split(",")
             priorbox=list(map(float,arr))
    priorbox=np.reshape(np.array(priorbox),(2,outlen))

    outputs = rknn.inference(inputs=[img])#,data_format="nchw",data_type="float32"

    print("pb:",priorbox.shape,priorbox)
    print("loc:",outputs[0].shape,outputs[0])
    print("conf:",outputs[1].shape,outputs[1])    

    NUM_RESULTS=outlen//4
    NUM_CLASSES=2
    box_priors= priorbox[0].reshape((NUM_RESULTS,4))
    box_var   = priorbox[1].reshape((NUM_RESULTS,4))
    loc =  outputs[0].reshape((NUM_RESULTS, 4))
    conf = outputs[1].reshape((NUM_RESULTS, NUM_CLASSES))

    #compute softmax
    conf = [[x/(x+y),y/(x+y)] for x,y in np.exp(conf)]

    # Post Process
    for i in range(0, NUM_RESULTS):

        pb = box_priors[i]
        lc = loc[i]
        var= box_var[i]

        pb_w = pb[2] - pb[0]
        pb_h = pb[3] - pb[1]
        pb_cx = (pb[0] + pb[2]) * 0.5;
        pb_cy = (pb[1] + pb[3]) * 0.5;

        bbox_cx = var[0] * lc[0] * pb_w + pb_cx;
        bbox_cy = var[1] * lc[1] * pb_h + pb_cy;
        bbox_w = math.exp(var[2] * lc[2]) * pb_w;
        bbox_h = math.exp(var[3] * lc[3]) * pb_h;

        xmin = bbox_cx - bbox_w * 0.5;
        ymin = bbox_cy - bbox_h * 0.5;
        xmax = bbox_cx + bbox_w * 0.5;
        ymax = bbox_cy + bbox_h * 0.5;

        xmin *= 300 #input width
        ymin *= 300 #input height
        xmax *= 300 #input width
        ymax *= 300 #input height

        score = conf[i][1];

        if score > 0.9:
            print("score:",score)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),(0, 0, 255), 3)

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    plt.show()

    print("ssd finished")

if __name__=="__main__":
    img_path="test.jpeg"
    priorbox_path="priorbox_flatten.txt"
    run_ssd(img_path,priorbox_path)

