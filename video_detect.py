#coding=utf-8
from compiler.ast import flatten
import caffe
import numpy as np
# import pymysql
import os
import os.path as osp
from PIL import Image
# import  MySQLdb as mdb
import struct
import time
# from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import cv2


def imgfeature (img):
    # 图片预处理设置
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    transformer.set_mean('data', np.array([63.7, 79.3, 136.1]))  # 减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

    im = caffe.io.load_image(img)  # 加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

    # 执行测试
    out = net.forward()

    file = open(labels_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    # 得到网络的最终输出结果
    loc = net.blobs['detection_out'].data[0][0]

    # labels = np.loadtxt(labels_filename, str, delimiter='\t')  # 读取类别名称文件
    #
    # prob = net.blobs['softmax'].data[0].flatten()  # 取出最后一层（Softmax）属于某个类别的概率值，并打印
    # print (prob)
    # order = prob.argsort()[1]  # 将概率值排序，取出最大值所在的序号
    # print( 'the class is:', labels[order] ) # 将该序号转换成对应的类别名称，并打印
    return loc,im,labelmap

caffe.set_device(0)
caffe.set_mode_gpu()

if __name__ == '__main__':
#加载model
    # root = '//172.16.200.5/Data Share/MIP/SSD-Gastric/'  # 根目录
    root='/home/yangyh/caffe_root/caffe/img_detect_try/'
    deploy = root + 'deploy.prototxt'  # deploy文件
    caffe_model = root + 'Gastric_SSD_300x300_iter_120000.caffemodel'  # 训练好的 caffemodel
    labels_file = root + 'labelmap_gastric.prototxt'  # 类别名称文件，将数字标签转换回类别名称
    # mean_file = root + 'mean.npy'
    # mean=np.load(mean_file)
    net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network

    videoroot='./20150104001030008.jpg'
    cam = cv2.VideoCapture(videoroot)
    frame_count = 1
    success = True
    while(success):
        success, frame = cam.read()
        # image = skimage.img_as_float(frame).astype(np.float32)
        params = []
        params.append(cv.CV_IMWRITE_PXM_BINARY)
        params.append(1)
        cv2.imwrite('./output/' + "_%d.jpg" % frame_count, frame, params)
        frame_count = frame_count + 1
    cam.release()
    image_root='./output/'
    images = os.listdir(image_root)
    for each in images:
        strimgroot=image_root+each
        loc, im, labelmap = imgfeature(strimgroot)
        vis = im.copy()
        confidence_threshold = 0.5
        for l in range(len(loc)):
            if loc[l][2] >= confidence_threshold:
                # 目标区域位置信息
                xmin = int(loc[l][3] * im.shape[1])
                ymin = int(loc[l][4] * im.shape[0])
                xmax = int(loc[l][5] * im.shape[1])
                ymax = int(loc[l][6] * im.shape[0])
                # 画出目标区域
                cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)
                # 确定分类类别
                class_name = labelmap.item[int(loc[l][1])].display_name
                cv2.putText(vis, class_name, (xmin, ymax), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
                xx1=vis[:,:,0]*255
                xx2=vis[:,:,1]*255
                xx3=vis[:,:,2]*255
                r=Image.fromarray(xx1).convert('L')
                g=Image.fromarray(xx2).convert('L')
                b=Image.fromarray(xx3).convert('L')
                image=Image.merge('RGB',(r,g,b))
                image.save('./input/%d.jpg'%frame_count)

    img_root='./input/'
    videos = os.listdir(img_root)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,24,(1920,1080))#最后一个是保存图片的尺寸
    for i in range(len(videos)):
        frame = cv2.imread(img_root+str(i+1)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()



