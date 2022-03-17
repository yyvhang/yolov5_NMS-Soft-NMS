# 删除标注中height或者widh为0的xml和其对应的image
# 因为yolo格式的标注会对label做归一化，所以height和weight字段不能为0
from __future__ import annotations
import xml.etree.ElementTree as ET
import os

from cv2 import split

def delete(xml_path, image_path):
    xml_list = os.listdir(xml_path)
    for xml in xml_list:
        annotations_path = os.path.join(xml_path,xml)
        #print(annotations_path)
        file = open(annotations_path)
        tree = ET.parse(file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        if(w==0 or h==0):
            image_name = xml.split('.')[0]
            print('delete_label:{}'.format(annotations_path))
            os.remove(annotations_path)
            Image_path = os.path.join(image_path,image_name+'.jpg')
            print('delete_image:{}'.format(Image_path))
            os.remove(Image_path)

xml_path = '/home/project/Yolov5/yolov5/VOCdevkit/VOC2007/Annotations/'
image_path = '/home/project/Yolov5/yolov5/OCdevkit/VOC2007/JPEGImages/'
delete(xml_path, image_path)
