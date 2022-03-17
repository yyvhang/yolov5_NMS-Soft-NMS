import os
from tkinter import image_names
##将没有标注的图像删除，使图像和标签满足一一对应
def delete_img(xml_path, Image_path):
    xml_list = os.listdir(xml_path)
    img_list = []
    for xml in xml_list:
        image_name = xml.split('.')[0]
        image_path = image_name+'.jpg'
        img_list.append(image_path)

    all_image_list = os.listdir(Image_path)
    #delete_img_list = []
    for img in all_image_list:
        if(img not in img_list):
            delete_path = os.path.join(Image_path, img)
            #delete_img_list.append(delete_path)
            os.remove(delete_path)
    


xml_path = '/home/project/Yolov5/yolov5/VOCdevkit/VOC2007/Annotations/'  #存放xml标注的路径
Img_path = '/home/project/Yolov5/yolov5/VOCdevkit/VOC2007/JPEGImages/'   #存放图像的路径
delete_img(xml_path, Img_path)

