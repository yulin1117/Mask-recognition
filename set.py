import os
import numpy as np
from pathlib import Path
from xml.dom.minidom import parse
from shutil import copyfile

FILE_ROOT = "/Users/yulin/yolo/kaggle_face_mask/"
IMAGE_PATH = FILE_ROOT + "images"  
ANNOTATIONS_PATH = FILE_ROOT + "annotations"
DATA_ROOT = "/Users/yulin/yolo/yolov7/Dataset/"
LABELS_ROOT = DATA_ROOT + "FaceMask/labels"
IMAGES_ROOT = DATA_ROOT + "FaceMask/images"  

classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
#------------轉.txt已符合darknet格式才能使用yolo訓練-------------
def cord_converter(size, box):
    """
    convert xml annotation to darknet format coordinates
    :param size： [w,h]
    :param box: anchor box coordinates [upper-left x,uppler-left y,lower-right x, lower-right y]
    :return: converted [x,y,w,h]
    以下是一个例子说明这个过程：
    比如在一张 500x500 的图片中有一个边界框，其在 xml 文件中的坐标是：[50, 50, 150, 150]，则其宽度 w=150-50=100，高度 h=150-50=100，中心点位置x=50+(100/2)=100， y=50+(100/2)=100。然后对 x, y, w, h 进行归一化，x=x/dw=100/500=0.2, y=y/dh=100/500=0.2, w=w/dw=100/500=0.2, h=h/dh=100/500=0.2。所以该 bounding box 在 darknet 格式下的表示为 [0.2,0.2,0.2,0.2]。
    """
    #box的邊界框的左上角和右下角的座標
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))#正規化box的大小
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]#回傳box的中心點和長寬

#呼叫cord_converter正規劃座標後轉換成.txt檔案
def save_file(img_jpg_file_name, size, img_box):
    save_file_name = LABELS_ROOT + '/' + img_jpg_file_name + '.txt'
    print(save_file_name)
    file_path = open(save_file_name, "a+")
    for box in img_box:

        cls_num = classes.index(box[0])

        new_box = cord_converter(size, box[1:])

        file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")

    file_path.flush()
    file_path.close()

#把xml描述轉成imgbox list裡面存放多個物件，再存成txt檔案
def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '/' + img_xml_file + '.xml'
    print(img_path)

    dom = parse(img_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    img_size = root.getElementsByTagName("size")[0]
    objects = root.getElementsByTagName("object")
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    # print("img_name:", img_name)
    # print("image_info:(w,h,c)", img_w, img_h, img_c)
    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        # print("box:(c,xmin,ymin,xmax,ymax)", cls_name, x1, y1, x2, y2)
        img_jpg_file_name = img_xml_file + '.jpg'
        img_box.append([cls_name, x1, y1, x2, y2])
    # print(img_box)

    # test_dataset_box_feature(img_jpg_file_name, img_box)
    save_file(img_xml_file, [img_w, img_h], img_box)

#把ANNOTATIONS資料夾裡面的文件一個一個讀取做以上三步驟轉成.txt檔案
files = os.listdir(ANNOTATIONS_PATH)
for file in files:
    print("file name: ", file)
    file_xml = file.split(".")
    get_xml_data(ANNOTATIONS_PATH, file_xml[0])

#------------分成训练集、验证集和测试集-------------
import splitfolders

datapath = "/Users/yulin/yolo/yolov7/Dataset/FaceMask"
splitfolders.ratio(datapath, output="/Users/yulin/yolo/yolov7/Dataset/split_facemask", seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
#------------Create the Data YAML File-------------
# Download the Tiny model weights.
#!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
