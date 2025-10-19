#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import shutil

import cv2
import detectron2.utils.comm as comm
import torch.nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.visualizer import Visualizer

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class Register:
    """
    用于注册自己的数据集
        1. 首先保证自己的数据集标注是coco格式，就可以使用load_coco_json加载自己的数据集并转化为detectron的转悠数据格式
        2. 使用DatasetCatalog.register注册训练集和测试集
        3. 使用MetadataCatalog.get注册训练集和测试集的标注元数据
    """
    CLASS_NAMES = ['__background__', '0']  # 保留 background 类
    def __init__(self, dataset_root):
        self.CLASS_NAMES = Register.CLASS_NAMES or ['__background__', ]
        # 数据集路径
        self.DATASET_ROOT = dataset_root
        # ANN_ROOT = os.path.join(self.DATASET_ROOT, 'COCOformat')
        self.ANN_ROOT = self.DATASET_ROOT
        self.TRAIN_PATH = os.path.join(self.DATASET_ROOT, 'images/train')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'annotations/train.json')
        self.VAL_PATH = os.path.join(self.DATASET_ROOT, 'images/test')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'annotations/test.json')
        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            "my_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "my_val": (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(self, name=key, json_file=json_file, image_root=image_root)

    @staticmethod
    def register_dataset_instances(self, name, json_file, image_root):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """

        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get(name).set(json_file=json_file,
                                      image_root=image_root,
                                      evaluator_type="coco")

    def plain_register_dataset(self):
        """注册数据集和元数据"""
        # 训练集
        DatasetCatalog.register("my_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("my_train").set(thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                 evaluator_type='coco',  # 指定评估方式
                                                 json_file=self.TRAIN_JSON, image_root=self.TRAIN_PATH)
        # DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
        # 验证/测试集
        DatasetCatalog.register("my_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("my_val").set(thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                               evaluator_type='coco',  # 指定评估方式
                                               json_file=self.VAL_JSON, image_root=self.VAL_PATH)

    def checkout_dataset_annotation(self, name="my_val"):
        """
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
        # 查看训练集前一百张张图片的GT
        dataset_dicts = load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        print("\nThe number of the training set: {}\n".format(len(dataset_dicts)))
        if os.path.exists('out/train'):
            shutil.rmtree('out/train')
        os.mkdir('out/train')
        for i, d in enumerate(dataset_dicts, 0):
            # print(d)
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow('show', vis.get_image()[:, :, ::-1])
            cv2.imwrite('out/train/' + d["file_name"].split('/')[-1].split('.')[0] + '_withgt.jpg', vis.get_image()[:, :, ::-1])
            if i == 100:
                break
        # 查看测试集所有图片的GT
        dataset_dicts = load_coco_json(self.VAL_JSON, self.VAL_PATH)
        print("\nThe number of the test set: {}\n".format(len(dataset_dicts)))
        if os.path.exists('out/test'):
            shutil.rmtree('out/test')
        os.mkdir('out/test')
        for i, d in enumerate(dataset_dicts, 0):
            # print(d)
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow('show', vis.get_image()[:, :, ::-1])
            cv2.imwrite('out/test/' + d["file_name"].split('/')[-1].split('.')[0] + '_withgt.jpg', vis.get_image()[:, :, ::-1])
            # if i == 10:
            #     break


def main(args):
    cfg = setup(args)
    """
    一、注册自己的数据集
       使用detectron2训练自己的数据集，第一步要注册自己的数据集
    """
    data_Register = Register('/media/dell/codes/liyiming/CIL/datasets/' + 'SRSDD')
    data_Register.register_dataset()
    # data_Register.checkout_dataset_annotation()
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)
            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    """
    二、训练自己的数据集
        1. 在官方的tools/train_net.py上加上注册数据集部分
        2. 继承DefaultTrainer父类，定义Trainer，重写build_evaluator类方法，使得Trainer具有评估功能
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    args.num_gpus = 1
    args.config_file = 'configs/coco_supervision/faster_rcnn_R_50_FPN_CIL.yaml'
    args.opts = ['SOLVER.IMG_PER_BATCH_LABEL', '8', 'SOLVER.IMG_PER_BATCH_UNLABEL', '8']
    main(args)


