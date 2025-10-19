# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import os
import pickle
import queue
import sys
import time
import logging
from contextlib import ExitStack
from collections import deque
import cv2
import pandas as pd
import torch
from torchvision import transforms
from natsort import natsorted
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from typing import Any, Dict, List, Set, Union
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators, \
    DatasetEvaluator, print_csv_format, inference_context
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
from tqdm import tqdm
from ubteacher.distillation.distillation_loss import *
from ubteacher.exemplar_exploitation.exemplar_selection import *

from configs.lib_test import cal_f1score, DrawDetBoxOnImg_returnimg_multiclass, nms_cpu_ship, \
    DrawDetBoxOnImgCompareGT_returnimg_multiclass, label_color_list
from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.modeling.roi_heads.fast_rcnn import FocalLoss
from ubteacher.prototype.find_nearest_feats import find_nearest_feats
from ubteacher.solver.build import build_lr_scheduler
from configs.lib_data import *
from torchvision import datasets, models, transforms
CLASSES = ['Container', 'cell-container', 'Dredger', 'ore-oil', 'Fishing', 'LawEnforce']
num_classes = len(CLASSES)
folder = '/media/dell/codes/liyiming/CIL/otsu'
""" 测试阶段去除虚警操作 """
POST = True

""" 存储模块 """
training_instances = {'Container': 2048, 'cell-container': 87, 'Dredger': 258,
                      'ore-oil': 173, 'Fishing': 240, 'LawEnforce': 35}
exemplar_rehearsal = {'Container': deque(maxlen=307), 'cell-container': deque(maxlen=13), 'Dredger': deque(maxlen=38),
                      'ore-oil': deque(maxlen=25), 'Fishing': deque(maxlen=36), 'LawEnforce': deque(maxlen=5)}
# exemplar_rehearsal = {'Container': [], 'cell-container': [], 'Dredger': [],
#                       'ore-oil': [], 'Fishing': [], 'LawEnforce': []}
# prototypes = {'Container': [], 'cell-container': [], 'Dredger': [],
#                       'ore-oil': [], 'Fishing': [], 'LawEnforce': []}
# selected = 0.25
###################### 测试数据的路径 ######################
test_path = '/media/dell/DataSets/SRSDD-V1.0/big_img_new/'
test_img_path, test_ann_path = test_path + 'JPEGImages/', test_path + 'Annotations/'

###################### 与测试相关的一些参数 ######################
# config = test_config
SCORE_THRESH, NMS_THRESH, tp_iou_thr = 0.80, 0.5, 0.3
cropw, croph, stride, TEST_BATCH_SIZE, LINE_W = 512, 512, 256, 1, 2
SAVE_DET_TO_IMGS, SAVE_DET_TO_TXTS, with_str = True, False, True

###################### 与检测和测试结果相关的一些路径 ######################
test_model_root = 'out/'
model_root = test_model_root
globals()['save_path'], globals()['save_test_txt_file'], globals()['save_current_img_path'], globals()[
    'save_current_pesudo_img_path'], globals()['iter'] = "", "", "", "", 0

choiceQueue = queue.Queue()
for choice in ['student', 'teacher']:
    choiceQueue.put(choice)

############################# crop sub_images #############################
need_det_img_name = os.listdir(test_img_path)
need_det_img_name = natsorted(need_det_img_name, alg=ns.PATH)  # 待测试图像
crop_subimage_path = test_path + 'crop_subimages/' + str(cropw) + '_' + str(croph) + '_' + str(stride) + '/'
makepath(crop_subimage_path)


if os.listdir(crop_subimage_path) == []:
    for ori_img_name in need_det_img_name:
        im_name0 = ori_img_name.split('.')[0]
        start_det = time.time()
        # print(ori_img_name)
        ori_img_file = test_img_path + ori_img_name
        img = cv2.imread(ori_img_file)
        w = img.shape[1]
        h = img.shape[0]  # 原始测试大图的长和宽
        ## crop and det together
        det_restore = []
        num_cropw = int(math.floor((w - cropw) / stride + 1))
        num_croph = int(math.floor((h - croph) / stride + 1))
        num_sub = (num_cropw + 1) * (num_croph + 1)
        for i in range(num_cropw + 1):
            for j in range(num_croph + 1):
                xmi = i * stride
                ymi = j * stride
                xmx = xmi + cropw
                ymx = ymi + croph
                if (xmx > w):
                    xmi = w - cropw
                    xmx = w
                if (ymx > h):
                    ymi = h - croph
                    ymx = h
                img_crop = img[ymi:ymx, xmi:xmx, :]
                cv2.imwrite(crop_subimage_path + im_name0 + '_' + str(xmi) + '_' + str(ymi) + '.png', img_crop)
sub_images = os.listdir(crop_subimage_path)
sub_images = natsorted(sub_images, alg=ns.PATH)


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = cls.inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def inference_on_dataset(cls, model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]):
        confusion_matrix_AllImg = np.zeros(shape=[num_classes + 1, num_classes + 1])
        missclass_matrix_AllImg = np.zeros(shape=num_classes)
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        evaluator.reset()
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())
            for idx, inputs in enumerate(data_loader):
                ori_img_name = inputs[0]['file_name'].split('/')[-1]
                im_name0 = inputs[0]['file_name'].split('/')[-1].split('.')[0]
                print(ori_img_name + '--------------------------------')
                ori_img_file = test_img_path + ori_img_name
                ori_xml_file = test_ann_path + im_name0 + '.xml'
                img = cv2.imread(ori_img_file)
                w = img.shape[1]
                h = img.shape[0]  # 原始测试大图的长和宽
                num_cropw = int(math.floor((w - cropw) / stride + 1))
                num_croph = int(math.floor((h - croph) / stride + 1))
                num_sub = num_cropw * num_croph

                det_restore = []
                images_path = []
                xoyos = []
                num_sub_i = 0
                batch_sample = 0
                det_res = []
                for sub_image in sub_images:  # 遍历所有的子图像
                    if sub_image.split('_')[0] != im_name0:
                        continue
                    sub_image_path = crop_subimage_path + sub_image
                    xmi = float(sub_image.split('.')[0].split('_')[1])
                    ymi = float(sub_image.split('.')[0].split('_')[2])
                    images_path.append(sub_image_path)  # 保存子图像路径
                    xoyos.append([xmi, ymi])  # 保存这些小图在大图上的坐标位置
                    batch_sample += 1
                    num_sub_i += 1
                    subs_det_res0 = []
                    if (batch_sample == TEST_BATCH_SIZE) or (num_sub_i == num_sub):
                        original_img = Image.open(sub_image_path)
                        # from pil image to tensor, do not normalize image
                        data_transform = transforms.Compose([transforms.ToTensor(),
                                                             transforms.Resize(800)
                                                             ])
                        img2 = data_transform(original_img)

                        inputs2 = [{'file_name': sub_image_path, 'height': croph, 'width': cropw, 'image_id': None,
                                    'image': img2 * 255}]
                        outputs = model(inputs2)  # 加载子图，开始测试

                        pred_class_id = getattr(outputs[0]['instances'], 'pred_classes').to("cpu").numpy()
                        scores = getattr(outputs[0]['instances'], 'scores').to("cpu").numpy()
                        bboxes = getattr(outputs[0]['instances'], 'pred_boxes').tensor.to("cpu").numpy()

                        if len(bboxes) == 0:
                            subs_det_res0.append([])
                        else:
                            inds = scores > SCORE_THRESH
                            pred_class_id = pred_class_id[inds]
                            scores = scores[inds]
                            bboxes = bboxes[inds, :]

                            pred_class_id = np.expand_dims(pred_class_id, 1)
                            scores = np.expand_dims(scores, 1)
                            class_and_scores = np.hstack((pred_class_id, scores))
                            det_res0 = np.hstack((class_and_scores, bboxes))
                            det_res0 = det_res0.tolist()
                            subs_det_res0.append(det_res0)

                        for sub_i in range(batch_sample):
                            for box in subs_det_res0[sub_i]:
                                box[2] += xoyos[sub_i][0]
                                box[3] += xoyos[sub_i][1]
                                box[4] += xoyos[sub_i][0]
                                box[5] += xoyos[sub_i][1]
                                det_restore.append(box)
                        images_path = []
                        xoyos = []
                        batch_sample = 0
                # print('Number of cropped subimgs:', num_sub)

                if det_restore != []:
                    # nms 必须需要，因为目前的策略是测试小图遍历
                    det_res_array = np.array(det_restore)
                    index = nms_cpu_ship(det_res_array, NMS_THRESH)
                    # print('detected bbox :%d' % len(index))
                    det_restore_nms = []
                    for index_i in index:
                        det_restore_nms.append(det_restore[index_i])

                    det_restore_res = np.array(det_restore_nms)  # 最终的检测结果
                    # compare with gt
                    bbox_gt = ReadXML(ori_xml_file)
                    det_res_img0, confusion_matrix_SingleImg, missclass_matrix_SingleImg = DrawDetBoxOnImgCompareGT_returnimg_multiclass(
                        img, det_restore_res, bbox_gt, LINE_W, tp_iou_thr, CLASSES, with_str)
                    confusion_matrix_AllImg += confusion_matrix_SingleImg
                    missclass_matrix_AllImg += missclass_matrix_SingleImg
                    if SAVE_DET_TO_IMGS:
                        det_res_img1 = DrawDetBoxOnImg_returnimg_multiclass(img, det_restore_res, LINE_W, CLASSES,
                                                                            with_str)
                        cv2.imwrite(globals()['save_current_img_path'] + im_name0 + '.png', det_res_img1)
                        cv2.imwrite(globals()['save_current_img_path'] + im_name0 + '_withgt.png', det_res_img0)
                    if SAVE_DET_TO_TXTS:
                        save_file = globals()['current_model_txt_path'] + im_name0 + '.txt'
                        with open(save_file, 'w') as f_img:
                            for bbox in det_res:
                                label = int(bbox[0])
                                label_str = CLASSES[label]
                                f_img.write(
                                    '{:s} {:.3f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(label_str, bbox[1], bbox[2],
                                                                                       bbox[3], bbox[4], bbox[5]))
                else:
                    missclass_matrix_SingleImg = np.zeros(shape=num_classes)
                    confusion_matrix_SingleImg = np.zeros(shape=[num_classes + 1, num_classes + 1])
                    bbox_gt = ReadXML(ori_xml_file)
                    for key in bbox_gt:
                        class_id = CLASSES.index(key)
                        for box in bbox_gt[key]:
                            confusion_matrix_SingleImg[class_id, -1] += 1
                    confusion_matrix_AllImg += confusion_matrix_SingleImg
                    if SAVE_DET_TO_TXTS:
                        save_file = globals()['current_model_txt_path'] + ori_img_name + '.txt'
                        with open(save_file, 'a+') as f_img:
                            f_img.write('0 0 0 0 0 0')

                # cal f1score
                RESULT_DICT_SingleImg, mean_f1_score_SingleImg, = cal_f1score(confusion_matrix_SingleImg, CLASSES,
                                                                              missclass_matrix_SingleImg)
                for label_str in RESULT_DICT_SingleImg:
                    NUM_DET = RESULT_DICT_SingleImg[label_str]['NUM_DET']
                    NUM_GT = RESULT_DICT_SingleImg[label_str]['NUM_GT']
                    NUM_FALSE = RESULT_DICT_SingleImg[label_str]['NUM_FALSE']
                    NUM_MISS = RESULT_DICT_SingleImg[label_str]['NUM_MISS']
                    precision = RESULT_DICT_SingleImg[label_str]['Precision']
                    recall = RESULT_DICT_SingleImg[label_str]['Recall']
                    f1score = RESULT_DICT_SingleImg[label_str]['F1score']
                    # print(label_str)
                    # print('NUM_DET:%d,NUM_GT:%d,NUM_FALSE:%d,NUM_MISS:%d' % (NUM_DET, NUM_GT, NUM_FALSE, NUM_MISS))
                    # print('precision:%.4f;recall:%.4f;F1-score:%.4f' % (precision, recall, f1score))
                # print('mean f1score 0f {}: {}'.format(ori_img_name, mean_f1_score_SingleImg))

            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print("\t===> 整体检测识别的混淆矩阵 <===")
        print(confusion_matrix_AllImg)
        NUM_GT = np.sum(confusion_matrix_AllImg[:-1, :], 1)
        NUM_DET = np.sum(confusion_matrix_AllImg[:-1, :-1], 1)
        NUM_CORRECT = np.diag(confusion_matrix_AllImg)[:-1]
        NUM_FALSE = confusion_matrix_AllImg[-1, :][:-1]

        TPR = np.sum(NUM_DET) / np.sum(NUM_GT)
        FPR = np.sum(NUM_FALSE) / (np.sum(NUM_DET) + np.sum(NUM_FALSE))
        ACCURACY = np.sum(NUM_CORRECT) / np.sum(NUM_DET)
        F1_SCORE = (2 * TPR * (1 - FPR)) / (TPR + (1 - FPR))
        TPR_unknown = np.sum(NUM_DET[-1]) / np.sum(NUM_GT[-1])
        ACCURACY_unknown = np.sum(NUM_CORRECT[-1]) / np.sum(NUM_DET[-1])
        print("===>检测率:\t\t%.2f%%\t识别准确率:\t   %.2f%%\t虚警率: %.2f%%\tF1分数: %.2f%%" % (
            100 * TPR, 100 * ACCURACY, 100 * FPR, 100 * F1_SCORE))
        print("===>未知类检测率: %.2f%%\t未知类识别准确率: %.2f%%" % (100 * TPR_unknown, 100 * ACCURACY_unknown))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Unbiased Teacher Trainer
class UBTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create a student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create a teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    globals()['iter'] = self.iter
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def remove_new_label(self, label_data):
        for label_datum in label_data:
            class_name = MetadataCatalog.get("my_train").thing_classes
            new_class_name = self.cfg.DATALOADER.NAME_NEW_CLASSES
            new_class_index = [class_name.index(b) for b in new_class_name if b in class_name]
            instances = label_datum['instances']
            fields = instances._fields
            gt_boxes = fields['gt_boxes']
            gt_classes = fields['gt_classes']
            # 判断gt_classes中，哪些类别是新类
            mask = torch.isin(gt_classes, torch.tensor(new_class_index))
            filtered_gt_classes = gt_classes[~mask]
            filtered_gt_boxes = gt_boxes[~mask]

            instances._fields['gt_boxes'] = filtered_gt_boxes
            instances._fields['gt_classes'] = filtered_gt_classes
        return label_data

    def remove_old_label(self, label_data):
        for label_datum in label_data:
            class_name = MetadataCatalog.get("my_train").thing_classes
            old_class_name = self.cfg.DATALOADER.NAME_OLD_CLASSES
            old_class_index = [class_name.index(b) for b in old_class_name if b in class_name]
            instances = label_datum['instances']
            fields = instances._fields
            gt_boxes = fields['gt_boxes']
            gt_classes = fields['gt_classes']
            # 判断gt_classes中，哪些类别是旧类
            mask = torch.isin(gt_classes, torch.tensor(old_class_index))
            filtered_gt_classes = gt_classes[~mask]
            filtered_gt_boxes = gt_boxes[~mask]

            instances._fields['gt_boxes'] = filtered_gt_boxes
            instances._fields['gt_classes'] = filtered_gt_classes
        return label_data

    def add_label(self, new_data, exemplar_rehearsal, mode='mixup'):
        """
        将旧目标放置于新数据中
        """
        if mode not in ['replay', 'mixup']:
            raise ValueError(f'mode {mode} not in [replay, mixup]')
        for data in new_data:
            if mode == 'replay':
                merge_instances(data, exemplar_rehearsal, OLD_CLASSES=self.cfg.DATALOADER.NAME_OLD_CLASSES, mode='both')
            if mode == 'mixup':
                merge_instances_mixup(data, exemplar_rehearsal, OLD_CLASSES=self.cfg.DATALOADER.NAME_OLD_CLASSES, mode='both')
        return

    def draw_pesudo_label(self, imgs, iter, back, root):
        for img in imgs:
            pesudo_labels = getattr(img['instances'], 'gt_classes').to("cpu").numpy()
            pesudo_scores = getattr(img['instances'], 'scores').to("cpu").numpy() if 'scores' in img[
                'instances'].get_fields().keys() else len(pesudo_labels) * [1.0]
            pesudo_bboxes = getattr(img['instances'], 'gt_boxes').to("cpu")
            cpimg = img['image'].cpu().numpy()
            cpimg = cpimg.transpose(1, 2, 0).copy()
            for index, pesudo_bbox in enumerate(pesudo_bboxes):
                label = pesudo_labels[index]
                score = pesudo_scores[index]
                xmin = int(pesudo_bbox[0])
                ymin = int(pesudo_bbox[1])
                xmax = int(pesudo_bbox[2])
                ymax = int(pesudo_bbox[3])
                bbox_color0 = label_color_list[int(label)]
                if (with_str):
                    display_txt = '%s: %.3f' % (CLASSES[int(label)], score)
                    cv2.putText(cpimg, display_txt, (xmin + 6, ymin + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color0,
                                1)
                cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), bbox_color0, 2)
            if not os.path.exists(os.path.join(globals()['save_current_pesudo_img_path'], str(iter), root)):
                makepath(os.path.join(globals()['save_current_pesudo_img_path'], str(iter), root))
            cv2.imwrite(os.path.join(globals()['save_current_pesudo_img_path'], str(iter)) + '/' + root + '/' + img[
                'image_id'] + back + '.png', cpimg)

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        self.model_teacher.eval()
        assert self.model_teacher.training == False
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        """ LUMO分别表示pure_task1, mixed_task1, pure_task2, mixed_task2；qk分别表示强增强与弱增强 """
        L_data_q, L_data_k, U_data_q, U_data_k, M_data_q, M_data_k, O_data_q, O_data_k = data
        data_time = time.perf_counter() - start

        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            """ 任务0训练过程 """
            if self.cfg.MODEL.WEIGHTS[-3:] != 'pth':
                self.model.train()
                assert self.model.training == True
                # 如果不是加载模型，则重新训练模型
                U_data_q, U_data_k = self.remove_new_label(U_data_q), self.remove_new_label(U_data_k)
                L_data_k.extend(L_data_q[:len(L_data_q) // 2])
                L_data_k.extend(U_data_k[:len(U_data_k) // 2])
                record_dict, _, _, _ = self.model(L_data_k, branch="supervised")
                # weight losses
                loss_dict = {}
                for key in record_dict.keys():
                    if key[:4] == "loss":
                        loss_dict[key] = record_dict[key] * 1
                losses = sum(loss_dict.values())
            else:
                """ 旧目标存储在exemplar rehearsal """
                self.model.eval()
                assert self.model.training == False, "[UBTeacherTrainer] model was changed to eval mode!"
                with torch.no_grad():
                    predictions = self.model(L_data_k, branch="supervised")
                # 对教师模型进行推理，返回与真实框IoU大于0.9且预测类别正确的预测框
                filter_predictions, filter_slices, indexes = filter_predictions_by_iou_and_class(predictions, L_data_k,
                                                                                        iou_threshold=0.9)
                for filter_prediction, filter_slice in zip(filter_predictions, filter_slices):
                    pred_class = CLASSES[filter_prediction._fields['pred_classes']]
                    exemplar_rehearsal[pred_class].append([filter_prediction, filter_slice])
                # with torch.no_grad():
                #     box_features = self.model(L_data_k, given_proposals=predictions, branch="prototype")
                # for filter_prediction, filter_slice, box_feat in zip(filter_predictions, filter_slices, box_features[indexes]):
                #     pred_class = CLASSES[filter_prediction._fields['pred_classes']]
                #     exemplar_rehearsal[pred_class].append([filter_prediction, filter_slice])
                #     prototypes[pred_class].append(box_feat.to('cpu'))
                return
        else:
            self.model.train()
            assert self.model.training == True
            """ 任务1训练过程 """
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # 初始化任务1的学习率，并将教师模型定义为旧类模型，学生模型定义为新类模型
                self.optimizer.param_groups[0]['lr'] = 0.002
                self._update_teacher_model(keep_rate=0.00)

            O_data_q, O_data_k = self.remove_old_label(O_data_q), self.remove_old_label(O_data_k)

            # """ Exemplar mining """
            # if self.cfg.SEMISUPNET.EXEMPLAR_MINING:
            #     exemplar_mining(self.model_teacher, O_data_k)

            M_data_k.extend(M_data_q[:len(M_data_q) // 2])
            M_data_k.extend(O_data_k[:len(O_data_k) // 2])
            if self.cfg.SEMISUPNET.EXEMPLAR_REPLAY and self.iter % 1000 == 0:
                show_image(M_data_k, 'replay', 'before', self.iter)

            """ Exemplar replay """
            if self.cfg.SEMISUPNET.EXEMPLAR_REPLAY:
                self.add_label(M_data_k, exemplar_rehearsal, mode='replay')

            if self.cfg.SEMISUPNET.EXEMPLAR_REPLAY and self.iter % 1000 == 0:
                show_image(M_data_k, 'replay', 'after', self.iter)

            with torch.no_grad():
                tea_feats, tea_RoI, tea_pred_instances, tea_predictions, tea_proposals_rpn = \
                    self.model_teacher(M_data_k, branch="teacher_guide")
            """ Supervised loss of the student model on the new data """
            record_dict, stu_feats, [stu_RoI, stu_pred_instances, stu_predictions], _ = \
                self.model(M_data_k, branch="stu_supervised", tea_proposals=tea_proposals_rpn)

            """ Scene-level knowledge distillation """
            if self.cfg.SEMISUPNET.SCENE_LEVEL:
                record_dict['kd_scene'] = scene_level_distillation_loss(tea_feats, stu_feats, M_data_k)

            """ Instance-level knowledge distillation """
            if self.cfg.SEMISUPNET.INSTANCE_LEVEL:
                record_dict['kd_instance'] = instance_level_distillation_loss(tea_RoI, stu_RoI)

            """ Logit-level knowledge distillation """
            if self.cfg.SEMISUPNET.LOGIT_LEVEL:
                record_dict['kd_logit'] = logits_level_distillation_loss(tea_predictions[0], stu_predictions[0],
                                                                         tea_predictions[1], stu_predictions[1])
            record_dict.update(record_dict)

            """ Weight losses """
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1.0
                elif key == "kd_scene":
                    loss_dict[key] = record_dict[key] * 1.0
                elif key == 'kd_instance':
                    loss_dict[key] = record_dict[key] * 1.0
                elif key == 'kd_logit':
                    loss_dict[key] = record_dict[key] * 1.0
                elif key == 'mix_up':
                    loss_dict[key] = record_dict[key] * 1.0
                else:
                    loss_dict[key] = record_dict[key] * 1.0
            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def test(cls, cfg, model, choice=None, evaluators=None):
        if choice == None:
            globals()['save_path'] = model_root + 'TEACHER_STUDENT/' + choiceQueue.get() + '/TEST_RES'
        elif (globals()['iter'] + 1) <= cfg.TEST.EVAL_PERIOD:
            globals()['save_path'] = model_root + 'BURN_IN/' + choice + '/TEST_RES'
        else:
            globals()['save_path'] = model_root + 'TEACHER_STUDENT/' + choice + '/TEST_RES'

        globals()['save_current_img_path'] = globals()['save_path'] + '/DET_RES_IMGS/'  # 保存当前的测试条件下的测试结果大图
        globals()['current_model_txt_path'] = globals()['save_path'] + '/DET_RES_TXTS/'
        makepath(globals()['save_path'])
        makepath(globals()['save_current_img_path'])
        makepath(globals()['current_model_txt_path'])

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = cls.inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def inference_on_dataset(cls, model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]):
        confusion_matrix_AllImg = np.zeros(shape=[num_classes + 1, num_classes + 1])
        missclass_matrix_AllImg = np.zeros(shape=num_classes)
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        evaluator.reset()
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())
            for idx, inputs in tqdm(enumerate(data_loader)):
                ori_img_name = inputs[0]['file_name'].split('/')[-1]
                im_name0 = inputs[0]['file_name'].split('/')[-1].split('.')[0]
                ori_img_file = test_img_path + ori_img_name
                ori_xml_file = test_ann_path + im_name0 + '.xml'
                img = cv2.imread(ori_img_file)
                w = img.shape[1]
                h = img.shape[0]  # 原始测试大图的长和宽
                num_cropw = int(math.floor((w - cropw) / stride + 1))
                num_croph = int(math.floor((h - croph) / stride + 1))
                num_sub = num_cropw * num_croph

                det_restore = []
                images_path = []
                xoyos = []
                num_sub_i = 0
                batch_sample = 0
                det_res = []
                for sub_image in sub_images:  # 遍历所有的子图像
                    if sub_image.split('_')[0] != im_name0:
                        continue
                    sub_image_path = crop_subimage_path + sub_image
                    xmi = float(sub_image.split('.')[0].split('_')[1])
                    ymi = float(sub_image.split('.')[0].split('_')[2])
                    images_path.append(sub_image_path)  # 保存子图像路径
                    xoyos.append([xmi, ymi])  # 保存这些小图在大图上的坐标位置
                    batch_sample += 1
                    num_sub_i += 1
                    subs_det_res0 = []
                    if (batch_sample == TEST_BATCH_SIZE) or (num_sub_i == num_sub):
                        original_img = Image.open(sub_image_path)
                        # from pil image to tensor, do not normalize image
                        data_transform = transforms.Compose([transforms.ToTensor(),
                                                             transforms.Resize(800)
                                                             ])
                        img2 = data_transform(original_img)
                        inputs2 = [{'file_name': sub_image_path, 'height': croph, 'width': cropw, 'image_id': None,
                                    'image': img2 * 255}]
                        outputs = model(inputs2)  # 加载子图，开始测试

                        pred_class_id = getattr(outputs[0]['instances'], 'pred_classes').to("cpu").numpy()
                        scores = getattr(outputs[0]['instances'], 'scores').to("cpu").numpy()
                        bboxes = getattr(outputs[0]['instances'], 'pred_boxes').tensor.to("cpu").numpy()

                        if len(bboxes) == 0:
                            subs_det_res0.append([])
                        else:
                            inds = scores > SCORE_THRESH
                            pred_class_id = pred_class_id[inds]
                            scores = scores[inds]
                            bboxes = bboxes[inds, :]

                            pred_class_id = np.expand_dims(pred_class_id, 1)
                            scores = np.expand_dims(scores, 1)
                            class_and_scores = np.hstack((pred_class_id, scores))
                            det_res0 = np.hstack((class_and_scores, bboxes))
                            det_res0 = det_res0.tolist()
                            subs_det_res0.append(det_res0)

                        for sub_i in range(batch_sample):
                            for box in subs_det_res0[sub_i]:
                                box[2] += xoyos[sub_i][0]
                                box[3] += xoyos[sub_i][1]
                                box[4] += xoyos[sub_i][0]
                                box[5] += xoyos[sub_i][1]
                                det_restore.append(box)
                        images_path = []
                        xoyos = []
                        batch_sample = 0
                if det_restore != []:
                    # nms 必须需要，因为目前的策略是测试小图遍历
                    det_res_array = np.array(det_restore)
                    index = nms_cpu_ship(det_res_array, NMS_THRESH)
                    det_restore_nms = []
                    for index_i in index:
                        det_restore_nms.append(det_restore[index_i])

                    det_restore_res = np.array(det_restore_nms)  # 最终的检测结果
                    """ 测试阶段后处理 """
                    if POST:
                        # 海陆分割得到的结果同样交集到mask掩膜矩阵
                        mask_path = os.path.join(folder, 'mid_' + ori_img_name)
                        mask = Image.open(mask_path)
                        mask = np.array(mask)
                        delete = []
                        for ii, box in enumerate(det_restore_res):
                            label, score = box[0], box[1]
                            # 剔除虚警杂波，防止新目标漏检/置信度得分极高则不处理，防止漏检
                            if (label not in [0, 1, 2]) or (score > 0.95):
                                continue
                            xmin, ymin, xmax, ymax = int(box[2]), int(box[3]), int(box[4]), int(box[5])
                            height, width = int(ymax - ymin), int(xmax - xmin)
                            expand_by = 15
                            # 大局部像素比例计算
                            new_xmin = max(xmin - expand_by, 0)
                            new_ymin = max(ymin - expand_by, 0)
                            new_xmax = min(xmax + expand_by, w)
                            new_ymax = min(ymax + expand_by, h)
                            p11 = mask[new_ymin: new_ymax, new_xmin:new_xmax].sum()
                            p12 = (new_xmax - new_xmin) * (new_ymax - new_ymin)
                            p1 = p11 / p12
                            # 小局部像素比例计算
                            p21 = mask[ymin:ymax, xmin:xmax].sum()
                            p22 = height * width
                            p2 = p21 / p22
                            # 局部区域像素变化不显著，可能是内陆区域
                            if np.abs(p1 - p2) < 0.05:
                                delete.append(ii)
                        det_restore_res = np.delete(det_restore_res, delete, axis=0)

                    if det_restore_res.shape[0] == 0:
                        confusion_matrix_SingleImg = np.zeros(shape=[num_classes + 1, num_classes + 1])
                        bbox_gt = ReadXML(ori_xml_file)
                        for key in bbox_gt:
                            class_id = CLASSES.index(key)
                            for box in bbox_gt[key]:
                                confusion_matrix_SingleImg[class_id, -1] += 1
                        confusion_matrix_AllImg += confusion_matrix_SingleImg
                        continue
                    # compare with gt
                    bbox_gt = ReadXML(ori_xml_file)
                    det_res_img0, confusion_matrix_SingleImg, missclass_matrix_SingleImg = DrawDetBoxOnImgCompareGT_returnimg_multiclass(
                        img, det_restore_res, bbox_gt, LINE_W, tp_iou_thr, CLASSES, with_str)
                    confusion_matrix_AllImg += confusion_matrix_SingleImg
                    missclass_matrix_AllImg += missclass_matrix_SingleImg
                    if SAVE_DET_TO_IMGS:
                        det_res_img1 = DrawDetBoxOnImg_returnimg_multiclass(img, det_restore_res, LINE_W, CLASSES,
                                                                            with_str)
                        cv2.imwrite(globals()['save_current_img_path'] + im_name0 + '.png', det_res_img1)
                        cv2.imwrite(globals()['save_current_img_path'] + im_name0 + '_withgt.png', det_res_img0)
                else:
                    confusion_matrix_SingleImg = np.zeros(shape=[num_classes + 1, num_classes + 1])
                    bbox_gt = ReadXML(ori_xml_file)
                    for key in bbox_gt:
                        class_id = CLASSES.index(key)
                        for box in bbox_gt[key]:
                            confusion_matrix_SingleImg[class_id, -1] += 1
                    confusion_matrix_AllImg += confusion_matrix_SingleImg

        print("\t===> 整体检测识别的混淆矩阵 <===")
        print(confusion_matrix_AllImg)
        print("===>\t\t相关指标\t\t<===")
        NUM_CORRECT = np.diag(confusion_matrix_AllImg)[:-1]
        NUM_DET = np.sum(confusion_matrix_AllImg[:-1, :], 1)
        NUM_PRE = np.sum(confusion_matrix_AllImg[:, :-1], 0)
        # 类平均指标
        mRecall = NUM_CORRECT / NUM_DET
        mPrecision = NUM_CORRECT / (NUM_PRE + 1e-6)
        mF1 = 2 * mRecall * mPrecision / (mRecall + mPrecision + 1e-6)
        df = pd.DataFrame({
            'Class': CLASSES + ['old_average', 'new_average', 'average'],
            'Precision': mPrecision.tolist() + [np.average(mPrecision[:3])] + [np.average(mPrecision[-3:])] + [
                np.average(mPrecision)],
            "Recall": mRecall.tolist() + [np.average(mRecall[:3])] + [np.average(mRecall[-3:])] + [np.average(mRecall)],
            "F1": mF1.tolist() + [np.average(mF1[:3])] + [np.average(mF1[-3:])] + [np.average(mF1)]
        })
        df.set_index('Class', inplace=True)
        print(df)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key] * 1
                if key[:2] == "kd":
                    loss_dict[key] = metrics_dict[key] * 0.1
            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = student_model_dict[key] * (1 - keep_rate) + value * keep_rate
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model, choice='student')
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher, choice='teacher')
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
