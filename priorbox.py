import numpy as np
import sys,os
import cv2
#set caffe-ssd path
caffe_root = '/home/di/workspace/caffe-ssd/build/install/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import time


#caffe.set_device(0)
#caffe.set_mode_gpu()

#net_file= 'MobileNetSSD_deploy.prototxt'
net_file= 'MobileNetSSD_deploy_truncated.prototxt'
caffe_model='MobileNetSSD_deploy10695.caffemodel'

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)

CLASSES = ('background','person')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(img_path):

    origimg = cv2.imread(img_path)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    start = time.time()
    out = net.forward()
    use_time=time.time() - start
    print("time="+str(round(use_time*1000,3))+"ms")
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        if conf[i] > 0.3:
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            cv2.rectangle(origimg, p1, p2, (0,255,0))
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

    cv2.imshow("SSD", origimg)

    cv2.waitKey(1) & 0xff
        #Exit if ESC pressed
    return True

def get_prior_box(img_path):

    origimg = cv2.imread(img_path)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    start = time.time()
    out = net.forward()

    #get priorbox
    priorbox=out["mbox_priorbox"]
    print priorbox,np.shape(priorbox)
    pb=priorbox.flatten()
    with open("./priorbox_flatten.txt",'w') as f:
         f.write(",".join(map(str,pb)))

    #for view priorbox
    pb=np.reshape(pb,(1917,8))
    with open("./priorbox.txt",'w') as f:
         f.write("")
    with open("./priorbox.txt",'a') as f:
         for line in pb:
             f.write(",".join(map(str,line))+",\n")


if __name__ == '__main__':

    get_prior_box("test.jpeg")

