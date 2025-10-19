# -*- coding: UTF-8 -*-
#!/usr/bin/python
import shutil
from shutil import copyfile
import random
import numpy as np
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import cv2
import os
import matplotlib.image as mpimg
from PIL import Image
from natsort import ns, natsorted
Image.MAX_IMAGE_PIXELS = None

#from LABEL_LIST import label_list

COLORS = [(255,0,0),     (0,255,0),     (0,0,255),     (0,255,255),   (255,0,255), (255,255,0),   (255,0,127),   (0,127,255),\
         (127,0,255),  (127,128,255), (255,128,127), (127,255,128), (127,51,255),  (255,51,127),(127,255,51),  (51, 51, 255), (51, 255, 51), (255, 51, 51), (0, 51, 255), (51, 255, 0), (255, 0, 51)]
         
color_white = (255, 255, 255)
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_blue = (255, 0, 0)
color_black = (0, 0, 0)
color_yellow = (0, 255, 255)
color_orange =  (0, 165, 255)
color_deepred = (255, 0, 255)
color_cyan = (0, 255, 255)
color_purple = (255, 0, 255)
COLOR_RIGHT = color_green
COLOR_MISS = color_blue
COLOR_FALSE = color_red
COLOR_MISCLS = color_red


def makepath(path):
    if not os.path.exists(path):
       os.makedirs(path)


def imread(image_file,type=1):
    image =  mpimg.imread(image_file)
    if(len(image.shape)==3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ##for matplotlib
    if not(image_file.split('.')[-1]=='tiff'):
        image = image * 255
    return image

def copyfiles(old_path,new_path):
    makepath(new_path)
    for file_ in os.listdir(old_path):
        copyfile(old_path+file_, new_path+file_)

def gen_list(file_path,txt_file,rand=True):
    if os.path.exists(txt_file):
        print('Error:' + txt_file + ' already exists, please delete it!')
    else:
        files = os.listdir(file_path)
        if rand:
            files1 = random.sample(files,len(files))
        else:
            files1 = sorted(files)
        with open(txt_file, "a") as f:
            for file_ in files1:
                f.write(file_.split('.')[0]+'\n')
        f.close()

def check_voc_dataset(voc_data_path,CHECK_N,label_list_all,rand=False):
    new_imgs_path = voc_data_path + 'JPEGImages/'
    new_xml_path = voc_data_path + 'Annotations/'
    save_imgs_path = voc_data_path + 'JPEGImages_GT/'
    makepath(save_imgs_path)
    if rand == False:
        img_files = natsorted(os.listdir(new_imgs_path), alg=ns.PATH)[:CHECK_N]
    else:
        img_files = os.listdir(new_imgs_path)
        img_files = random.sample(img_files, len(img_files))
        img_files = img_files[:CHECK_N]
    for img_name in img_files:
        image = cv2.imread(new_imgs_path + img_name)
        xml_name = img_name.split('.')[0] + '.xml'
        gt_bbox = ReadXML(new_xml_path + xml_name)
        img = DrawGTBoxOnImg_returnimg_with_label(image, gt_bbox, label_list_all, [(255,0,0),(255,0,0)], 2, with_str=False)
        cv2.imwrite(save_imgs_path + img_name, img)

def DrawGTBoxOnImg_returnimg_with_label(image,bbox,label_list,bbox_colors,line_w,with_str=True):
    cpimg = image.copy()
    for key in bbox:
        for box in bbox[key]:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            index = label_list.index(key)
            if with_str:
                cv2.putText(cpimg, key, (xmin+6, ymin + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_colors[index], 2)
            cv2.rectangle(cpimg, (xmin, ymin), (xmax,ymax), bbox_colors[index], line_w)
    return cpimg

##get object annotation bndbox loc start 
def ReadXML(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)-1
        x2 = int(BndBox.find('xmax').text)-1
        y2 = int(BndBox.find('ymax').text)-1
        BndBoxLoc=[x1,y1,x2,y2]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet


def name(namelen,i):
    stri = str(i)
    strilen = len(stri)
    out = ''
    for l in range(namelen - strilen):
        out=out+'0'
    out = out + stri
    return out

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def createXML(database,folder, ImgName, Path, img_shape, bbox, save_dir):
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = ImgName
    ET.SubElement(annotation, 'path').text = Path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = database
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_shape[1])
    ET.SubElement(size, 'height').text = str(img_shape[0])
    if(len(img_shape) == 3):
        ET.SubElement(size, 'depth').text = str(img_shape[2])
    else:
        ET.SubElement(size, 'depth').text = '1'
    ET.SubElement(annotation, 'segmented').text = '0'

    for key in bbox:
        for box in bbox[key]:
            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = key
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(box[0])+1)
            ET.SubElement(bndbox, 'ymin').text = str(int(box[1])+1)
            ET.SubElement(bndbox, 'xmax').text = str(int(box[2])+1)
            ET.SubElement(bndbox, 'ymax').text = str(int(box[3])+1)

    treeString = prettify(annotation)

    treeString = treeString[treeString.find('\n') + 1: treeString.rfind('\n')]
    with open(save_dir + ImgName.split('.')[0] + '.xml', 'w') as xml_file:
        xml_file.write(treeString)
        


def iou_one2all(box, all_bbox):
    xmins = all_bbox[:,0]
    ymins = all_bbox[:,1]
    xmaxs = all_bbox[:,2]
    ymaxs = all_bbox[:,3]
    areas_all_bbox = (xmaxs - xmins + 1) * (ymaxs - ymins + 1)
    over_xmin = np.maximum(box[0], xmins[:])
    over_ymin = np.maximum(box[1], ymins[:])
    over_xmax = np.minimum(box[2], xmaxs[:])
    over_ymax = np.minimum(box[3], ymaxs[:])
    areas = (box[2] - box[0]) * (box[3] - box[1])
    w = np.maximum(0.0, over_xmax - over_xmin + 1)
    h = np.maximum(0.0, over_ymax - over_ymin + 1)
    inter = w * h
    over_v = inter / (areas + areas_all_bbox[:] - inter)
    return over_v.max(),np.argmax(over_v)
    
    
