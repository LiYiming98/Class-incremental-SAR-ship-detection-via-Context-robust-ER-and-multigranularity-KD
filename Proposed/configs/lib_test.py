# -*- coding: UTF-8 -*-
#!/usr/bin/python
from __future__ import division
import numpy as np
import cv2
import scipy
from configs.lib_data import *
import math
import time
#import shapely
#from shapely.geometry import Polygon, MultiPoint

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

label_color_list = [color_orange, color_orange, color_orange, color_green, color_green, color_green]


def makepath(path):
    if not os.path.exists(path):
       os.makedirs(path)


def cropimg(img_file, crop_width, crop_height, stride, save_path):
    img_name = img_file.split('/')[-1]
    save_xoyo_path =save_path+ '/crop_xoyo'
    save_imgs_path =save_path+ '/crop_imgs'
    makepath(save_xoyo_path)
    makepath(save_imgs_path)
    t = 0
    result = {}
    result["img_name"] = []
    result["crop_name"] = []
    result["crop_xoyo"] = []

    img = cv2.imread(img_file)
    w = img.shape[1]
    h = img.shape[0]
    num_cropw = int(math.floor((w - crop_width) / stride + 1))
    num_croph = int(math.floor((h - crop_height) / stride + 1))

    with open(os.path.join(save_xoyo_path,'crop.txt'), "w") as f:
        for i in range(num_cropw + 1):
            for j in range(num_croph + 1):
                xmi = i * stride
                ymi = j * stride
                xmx = xmi + crop_width
                ymx = ymi + crop_height
                if (xmx > w):
                    xmi = w - crop_width
                    xmx = w
                if (ymx > h):
                    ymi = h - crop_height
                    ymx = h
                t = t + 1
                crop_name = str(t) + '.png'
                img_crop = img[ymi:ymx, xmi:xmx]
                result["img_name"].append(str(img_name))
                result["crop_name"].append(str(crop_name))
                result["crop_xoyo"].append([xmi, ymi])
                savefile = os.path.join(save_imgs_path, crop_name)
                cv2.imwrite(savefile, img_crop)
                f.write(img_name + ' ' + crop_name + ' ' + str(xmi) + ' ' + str(ymi))
    f.close()
    print('Number of cropped imgs:', t)
    with open(save_xoyo_path + '/crop_xoyo.pkl', 'wb') as pkl_f:
        pickle.dump(result, pkl_f, pickle.HIGHEST_PROTOCOL)


def restore_(det_res, crop_xoyo):
    det_restore = []
    for name in det_res['img_name']:
        name_index = list.index(crop_xoyo['crop_name'],name)
        name_index2 = list.index(det_res['img_name'],name)
        xoyo = crop_xoyo['crop_xoyo'][name_index]
        #print(name,name_index,name_index2,xoyo)
        for box in det_res['det_res'][name_index2]:
            box[2] = box[2] + xoyo[0]
            box[3] = box[3] + xoyo[1]
            box[4] = box[4] + xoyo[0]
            box[5] = box[5] + xoyo[1]
            det_restore.append(box)
    return det_restore


def restore(det_res_file,crop_file):
    with open(crop_file, 'r') as pkl_f:
        crop_xoyo = pickle.load(pkl_f)
    with open(det_res_file, 'r') as pkl_f2:
        det_res = pickle.load(pkl_f2)
    return restore_(det_res,crop_xoyo)

def restore2(det_res,crop_file):
    with open(crop_file, 'rb') as pkl_f:
        crop_xoyo = pickle.load(pkl_f)
    return restore_(det_res,crop_xoyo)



def Color(image):
    scipy.misc.imsave('./temp.png', image)
    im_gray = cv2.imread('./temp.png', cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    return im_color


def ATT_Color(attention_map4):
    attention_map4 = np.resize(attention_map4, [38, 38]) * 255
    max_att = attention_map4.max()
    min_att = attention_map4.min()
    attention_map4 = (attention_map4 - min_att) / (max_att - min_att)
    attention_map4 = cv2.resize(attention_map4, (300, 300))
    att_map = Color(attention_map4)
    return att_map

        
def iou_one2all_ori(box, all_bbox):
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


def iou_one2all(box, all_bbox, way=0):
    ###当way=0时，iou
    ### iou=重叠部分/并集

    ###当way=1 way=2时，iou_soft
    ### iou_soft=重叠部分/GTbbox
    ##当way=1时，主要计算是否漏警，将第i个GTbbox与所有检测bbox比较，重叠率＝重叠部分/GTbbox（前者bbox）
    ##当way=2时，主要计算是否虚警，将第i个检测bbox与所有GTbbox比较，重叠率=重叠部分/GTbbox（后者bbox）

    xmins = all_bbox[:,0]
    ymins = all_bbox[:,1]
    xmaxs = all_bbox[:,2]
    ymaxs = all_bbox[:,3]
    areas_all_bbox = (xmaxs - xmins + 1) * (ymaxs - ymins + 1)
    over_xmin = np.maximum(box[0], xmins[:])
    over_ymin = np.maximum(box[1], ymins[:])
    over_xmax = np.minimum(box[2], xmaxs[:])
    over_ymax = np.minimum(box[3], ymaxs[:])
    w = np.maximum(0.0, over_xmax - over_xmin + 1)
    h = np.maximum(0.0, over_ymax - over_ymin + 1)
    inter = w * h
    if(way==0):
        areas = (box[2] - box[0]) * (box[3] - box[1])
        over_v = inter / (areas + areas_all_bbox[:] - inter)
    elif(way==1):
        areas = (box[2] - box[0]) * (box[3] - box[1])
        over_v = inter / areas
    elif(way==2):
        over_v = inter / (areas_all_bbox[:])


    return over_v.max(),np.argmax(over_v)

def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=True):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

####################### one class #######################
####################### one class #######################
def DrawDetBoxOnImg_returnimg(image,bbox,line_w,label_list,with_str=False):
    cpimg = image.copy()
    for box in bbox:
        label = box[0]
        score = box[1]
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4])
        ymax = int(box[5])
        bbox_color0 = label_color_list[int(label)]
        if(with_str):
            display_txt = '%s: %.3f' % (label_list[int(label)], score)
            cv2.putText(cpimg, display_txt, (xmin+6, ymin + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,bbox_color0, 1)
        cv2.rectangle(cpimg, (xmin, ymin), (xmax,ymax), bbox_color0,line_w)
    return cpimg


def DrawDetBoxOnImgCompareGT_returnimg(image, bbox_det, bbox_gt, line_w, false_iou_th, miss_iou_th, label_list, with_str=False):
    cpimg = image.copy()
    NUM_DET = len(bbox_det)
    NUM_FALSE = 0
    NUM_MISS = 0
    NUM_GT = 0
    for box in bbox_det:
        label = box[0]
        score = box[1]
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4])
        ymax = int(box[5])
        bbox_color = label_color_list[int(label)]
        gt_bbox = np.array(bbox_gt[label_list[int(label)]])
        det2gt_iou, max_index = iou_one2all(box[2:],gt_bbox,way=2)
        if(det2gt_iou < false_iou_th):
            bbox_color = COLOR_FALSE
            NUM_FALSE += 1
        if(with_str):
            display_txt = '%s: %.3f' % (label_list[int(label)], score)
            cv2.putText(cpimg, display_txt, (xmin, ymin - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
        cv2.rectangle(cpimg, (xmin, ymin), (xmax,ymax), bbox_color,line_w)

    for key in bbox_gt:
        bbox_det_temp = []
        for bbox_det_box in bbox_det:
            if (label_list[int(bbox_det_box[0])] == key):
                bbox_det_temp.append(bbox_det_box[2:])
        bbox_det_arr = np.array(bbox_det_temp)
        if (len(bbox_det_arr) > 0):
            for box in bbox_gt[key]:
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                gt2det_iou,max_index = iou_one2all(box,bbox_det_arr,way=1)
                if(gt2det_iou<miss_iou_th):
                    NUM_MISS+=1
                    cv2.rectangle(cpimg, (xmin, ymin), (xmax,ymax), COLOR_MISS,line_w)
    for key in bbox_gt:
        NUM_GT+=len(bbox_gt[key])
    return cpimg,NUM_DET,NUM_GT,NUM_FALSE,NUM_MISS
####################### one class #######################
####################### one class #######################

####################### multi class #######################
####################### multi class #######################
def nms_cpu_multiclass(dets, thresh):
    idex = []
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, 1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    class_ids = dets[:, 0]
    differ_classes = set(class_ids.flatten().tolist())
    for class_id_i in differ_classes:
        idex_i = []
        idex_class_i = np.where(class_ids == class_id_i)[0]
        #idex_class_i = idex_class_i.tolist()
        x1_i = x1[idex_class_i]
        y1_i = y1[idex_class_i]
        x2_i = x2[idex_class_i]
        y2_i = y2[idex_class_i]
        score_i = scores[idex_class_i]
        area_i = areas[idex_class_i]

        order_i = score_i.argsort()[::-1]
        while order_i.size > 0:
            i = order_i[0]
            idex_i.append(i)
            xx1 = np.maximum(x1_i[i], x1_i[order_i[1:]])
            yy1 = np.maximum(y1_i[i], y1_i[order_i[1:]])
            xx2 = np.minimum(x2_i[i], x2_i[order_i[1:]])
            yy2 = np.minimum(y2_i[i], y2_i[order_i[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (area_i[i] + area_i[order_i[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order_i = order_i[inds + 1]
        idex += idex_class_i[idex_i].tolist()

    return idex


def DrawDetBoxOnImg_returnimg_multiclass(image, bbox, line_w, label_list, with_str=False):
    cpimg = image.copy()
    for box in bbox:
        label = box[0]
        score = box[1]
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4])
        ymax = int(box[5])
        bbox_color0 = label_color_list[int(label)]
        if (with_str):
            CLASSES = ['Container', 'cell-container', 'Dredger', 'ore-oil', 'Fishing', 'LawEnforce']
            display_txt = '%s: %.3f' % (CLASSES[int(label)], score)
            cv2.putText(cpimg, display_txt, (xmin + 6, ymin + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color0, 1)
        cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), bbox_color0, line_w)
    return cpimg


def DrawDetBoxOnImgCompareGT_returnimg_multiclass(image, bbox_det, bbox_gt, line_w, tp_iou_thr, label_list, with_str=False):
    cpimg = image.copy()
    num_classes = len(label_list)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    missclass_matrix = np.zeros(shape=num_classes)
    gt_bboxes = []
    gt_labels = []
    for key in bbox_gt:
        for bbox in bbox_gt[key]:
            gt_bboxes.append(bbox)
            gt_labels.append(label_list.index(key))
    num_gt = len(gt_labels)
    
    true_positives = np.zeros(num_gt)
    missclass_false_negtive = np.zeros(num_gt)
    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    result = np.array(bbox_det)
    class_ids = result[:, 0].astype(int)
    det_bboxes = result[:, 2:6]
    det_scores = result[:, 1]
    ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)  # num_det * num_gt
    # 依次计算每个bbx与GT的IOU (det_scores已按照socre从大到小排列)
    for i, score in enumerate(det_scores):
        xmin = int(det_bboxes[i][0])
        ymin = int(det_bboxes[i][1])
        xmax = int(det_bboxes[i][2])
        ymax = int(det_bboxes[i][3])
        # for each det, the max iou with all gts
        max_ious1 = np.max(ious, 1)  # np.shape(det_bboxes)[0]
        # for each det, which gt overlaps most with it
        max_ious_idx1 = np.argmax(ious, 1)  # np.shape(det_bboxes)[0]
        # 如果第i个候选框并未检测到任何目标，该bbx可以直接判断为FP虚警
        if max_ious1[i] <= tp_iou_thr:
            confusion_matrix[-1, class_ids[i]] += 1
            bbox_color = COLOR_FALSE
            # print("索引" + str(i) + "的bbx候选框为虚警")
        # 如果第i个候选框与多个GT框匹配
        # 准则：bbx候选框检测到多个目标的主体特征，此时仅考虑IOU最大的GT框 #
        else:
            det_match = np.sum(ious[i] > tp_iou_thr)  # 第i个候选框与det_match个GT框匹配
            # print("索引 " + str(i) + " 的bbx候选框与 " + str(det_match) + " 个GT框建立了匹配关系。经判断，", end='')
            # 判断第i个bbx候选框与IOU最大的GT框的关系 #
            # 如果IOU最大的GT框已建立匹配关系，则第i个bbx候选框为FP虚警
            # 注意：如果令bbx候选框与IOU次最大的GT框建立匹配关系，
            #      存在次最大GT框与多个bbx候选框匹配的可能，导致次最大GT框无法与IOU最大的候选框建立匹配关系的风险 #
            if true_positives[max_ious_idx1[i]] != 0:
                confusion_matrix[-1, class_ids[i]] += 1
                bbox_color = COLOR_FALSE
                # print("索引" + str(i) + "的bbx候选框为虚警")
            elif class_ids[i] == gt_labels[max_ious_idx1[i]]:
                true_positives[max_ious_idx1[i]] += 1
                confusion_matrix[gt_labels[max_ious_idx1[i]], class_ids[i]] += 1
                bbox_color = label_color_list[class_ids[i]]
                # print("索引" + str(i) + "的bbx候选框为正确分类")
            else:
                true_positives[max_ious_idx1[i]] += 1
                confusion_matrix[gt_labels[max_ious_idx1[i]], class_ids[i]] += 1
                missclass_false_negtive[max_ious_idx1[i]] += 1
                bbox_color = COLOR_MISCLS
                # print("索引" + str(i) + "的bbx候选框为错误分类")

        if (with_str):
            display_txt = '%s: %.3f' % (label_list[class_ids[i]], score)
            cv2.putText(cpimg, display_txt, (xmin, ymin - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
        cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), bbox_color, line_w)

    # 绘制大图漏检框以及统计漏检number
    for num_tp, gt_label, gt_bbox in zip(true_positives, gt_labels, gt_bboxes):
        if num_tp == 0:  # FN 漏检
            xmin = int(gt_bbox[0])
            ymin = int(gt_bbox[1])
            xmax = int(gt_bbox[2])
            ymax = int(gt_bbox[3])
            bbox_color = COLOR_MISS
            if (with_str):
                display_txt = '%s: %.3f' % (label_list[class_ids[i]], score)
                cv2.putText(cpimg, display_txt, (xmin, ymin - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
            cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), bbox_color, line_w)
            confusion_matrix[gt_label, -1] += 1

    for num_fn_missclass, gt_label, gt_bbox in zip(missclass_false_negtive, gt_labels, gt_bboxes):
        if num_fn_missclass != 0:  # FN
            missclass_matrix[gt_label] += 1

    return cpimg, confusion_matrix, missclass_matrix

def cal_f1score(confusion_matrix, label_list, missclass_matrix):
    mean_f1score = 0
    RESULT_DICT = {}
    num_class = confusion_matrix.shape[0] - 1 # delete bg
    for class_id in range(num_class):
        num_gt = int(confusion_matrix[class_id, :].sum())
        if num_gt != 0:
            label = label_list[class_id]
            if label not in RESULT_DICT:
                RESULT_DICT[label] = {'Precision': 0, 'Recall': 0, 'F1_score': 0, 'NUM_DET': 0,
                                      'NUM_GT': 0, 'NUM_FALSE': 0, 'NUM_MISS': 0, 'NUM_MISSCLASS': 0}
            num_det = int(confusion_matrix[:, class_id].sum())
            tp = confusion_matrix[class_id][class_id]
            fp = num_det - tp
            fn = num_gt - tp
            fn_missclass = missclass_matrix[class_id]
            if tp == 0:
                precision = 0
                recall = 0
                f1score = 0
                mean_f1score += f1score
                udr = fn_missclass/fn
                udp = 0
            else:
                precision = tp / num_det
                recall = tp / num_gt
                f1score = (2 * precision * recall) / (precision + recall)
                mean_f1score += f1score
                udr = (tp+fn_missclass)/(tp+fn)
                udp = tp/(tp+fn_missclass)
            RESULT_DICT[label]['Precision'] = precision
            RESULT_DICT[label]['Recall'] = recall
            RESULT_DICT[label]['F1score'] = f1score
            RESULT_DICT[label]['NUM_DET'] = num_det
            RESULT_DICT[label]['NUM_GT'] = num_gt
            RESULT_DICT[label]['NUM_FALSE'] = fp
            RESULT_DICT[label]['NUM_MISS'] = fn
            RESULT_DICT[label]['NUM_MISSCLASS'] = fn_missclass
            RESULT_DICT[label]['UDP'] = udp
            RESULT_DICT[label]['UDR'] = udr
    mean_f1score /= len(RESULT_DICT)
    return RESULT_DICT, mean_f1score
####################### multi class #######################
####################### multi class #######################

def nms_cpu_ship(dets, thresh):
    # 获取当前目标类别下所有矩形框坐标bbx和置信度confidence
    x1, y1, x2, y2 = dets[:, 2], dets[:, 3], dets[:, 4], dets[:, 5]
    scores = dets[:, 1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对当前目标类别下所有的bbx的confidence进行从高到低排序，order记录索引信息
    order = scores.argsort()[::-1]
    idex = []   # 用来存放最大confidence对应的bbx索引
    # 依次从confidence从高到低遍历bbx，移除所有与该矩形框IOU值大于threshold的矩形框
    while order.size > 0:
        i = order[0]
        idex.append(i)  # 保留当前最大confidence对应的bbx索引
        # 获取所有与当前bbx的交集对应的左上角与右下角坐标，并计算IOU（注意这里是同时计算一个bbx与其他所有bbx的IOU）
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 当order.size==1时，下面的计算结果都为np.array([]),不影响最终结果
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        indexs = np.where(iou <= thresh)[0] + 1  # 获取保留下来的索引（因为没有计算与本身的IOU，所以索引相差1，需要加上）
        order = order[indexs]
    return idex
        
