# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import random

import numpy as np
import operator
import json
import torch.utils.data
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from ubteacher.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)
from typing import Dict, List
from tqdm import tqdm
from configs.lib_data import *

"""
This file contains the default logic to build a dataloader for training or testing.
"""

def divide_pure_and_mixed(dataset_dicts, ID, OOD):
    dicts = {"pure-ID": [],
             "pure-OOD": [],
             "mixed": [],
             "all": []
             }
    for index, dataset in enumerate(dataset_dicts):
        ID_OOD = [0, 0]
        dicts['all'].append(index)
        # 统计该图片目标中ID与OOD instances的数量
        for BBOX_GT in dataset['annotations']:
            ID_OOD[int(BBOX_GT['category_id'] / OOD[0])] += 1

        if ID_OOD[0] != 0 and ID_OOD[1] == 0:
            dicts["pure-ID"].append(index)
        elif ID_OOD[0] == 0 and ID_OOD[1] != 0:
            dicts['pure-OOD'].append(index)
        else:
            dicts['mixed'].append(index)

    return dicts


def write_to(SRSDD_random_idx, dicts, sup_percent):
    num_all = len(dicts['pure-ID'])
    # 写入 sup_percent% labeled set
    num_label = int(sup_percent / 100.0 * num_all)
    random.seed(1234)
    data = {
        "0": random.sample(dicts['pure-ID'], k=num_label),
        "1": random.sample(dicts['pure-ID'], k=num_label),
        "2": random.sample(dicts['pure-ID'], k=num_label),
        "3": random.sample(dicts['pure-ID'], k=num_label),
        "4": random.sample(dicts['pure-ID'], k=num_label),
    }
    SRSDD_random_idx[str(sup_percent)] = data


def generate_labeled_set(
    dataset_dicts,
    ID= [0, 1, 2, 3, 4],
    OOD = [5]
):
    SRSDD_random_idx = {}
    dicts = divide_pure_and_mixed(dataset_dicts, ID, OOD)

    for sup_percent in [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 1.0]:
        write_to(SRSDD_random_idx, dicts, sup_percent)

    return SRSDD_random_idx, len(dicts['pure-ID']), dicts


def divide_task1_task2(
    dataset_dicts, class_name, old_class, new_class, random_data_seed, random_data_seed_path
):
    """
    非语义隔离条件下的训练数据划分问题，每个批次的数据分为两部分：1）仅包含当前批次类别的pure_taski；2）包含其他批次类别的mixed_taski
    """
    pure_task1 = []
    mixed_task1 = []
    pure_task2 = []
    mixed_task2 = []
    print("Divide the training data into two tasks")
    for i in range(len(dataset_dicts)):
        category_ids = []
        for j in range(len(dataset_dicts[i]['annotations'])):
            category_ids.append(class_name[dataset_dicts[i]['annotations'][j]['category_id']])
        if not old_class.intersection(set(category_ids)):
            " 当前图像不包含任何旧类目标 "
            pure_task2.append(dataset_dicts[i])
        elif not new_class.intersection(set(category_ids)):
            " 当前图像不包含任何新类目标 "
            pure_task1.append(dataset_dicts[i])
        else:
            " 当前图像包含新旧类目标 "
            dataset_dicts_wo_new = dataset_dicts[i]
            dataset_dicts_wo_old = dataset_dicts[i]

            mixed_task1.append(dataset_dicts_wo_new)
            mixed_task2.append(dataset_dicts_wo_old)
    """ 避免图像复用，对mixed_task1和mixed_task2进行处理 """
    mixed_task1 = mixed_task1[:len(mixed_task1)//2]
    mixed_task2 = mixed_task2[len(mixed_task2)//2:]

    print("任务1类别：" + str(old_class) + " 任务1训练图像数：{}+{}".format(len(pure_task1), len(mixed_task1)))
    print("任务2类别：" + str(new_class) + " 任务2训练图像数：{} + {}".format(len(pure_task2), len(mixed_task2)))
    print("当前配置下没有图像复用问题")
    return pure_task1, mixed_task1, pure_task2, mixed_task2


def divide_label_unlabel(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    SRSDD_random_idx, num_all, dicts = generate_labeled_set(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    labeled_idx = np.array(SRSDD_random_idx[str(SupPercent)][str(random_data_seed)])
    assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    pure_ID_labeled_dicts = []
    pure_ID_unlabeled_dicts = []
    mixed_unlabeled_dicts = []
    pure_OOD_unlabeled_dicts= []
    pure_ID_labeled_idx = set(labeled_idx)
    pure_ID_unlabeled_idx = set(dicts['pure-ID']) - set(labeled_idx)
    mixed_unlabeled_idx = set(dicts['mixed'])
    # pure_OOD_unlabeled_idx = set(dicts['pure-OOD'])

    for i in range(len(dataset_dicts)):
        if i in pure_ID_labeled_idx:
            pure_ID_labeled_dicts.append(dataset_dicts[i])
        elif i in pure_ID_unlabeled_idx:
            pure_ID_unlabeled_dicts.append(dataset_dicts[i])
        elif i in mixed_unlabeled_idx:
            mixed_unlabeled_dicts.append(dataset_dicts[i])
        else:
            pure_OOD_unlabeled_dicts.append(dataset_dicts[i])

    return pure_ID_labeled_dicts, pure_ID_unlabeled_dicts, mixed_unlabeled_dicts, pure_OOD_unlabeled_dicts


# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    class_name = MetadataCatalog.get("my_train").thing_classes
    # Divide into task1 and task2 sets according to the class labels
    P1, M1, P2, M2 = divide_task1_task2(
            dataset_dicts,
            class_name,
            set(cfg.DATALOADER.NAME_OLD_CLASSES),
            set(cfg.DATALOADER.NAME_NEW_CLASSES),
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )
    ID_label_dataset = DatasetFromList(P1, copy=False)
    ID_unlabel_dataset = DatasetFromList(M1, copy=False)
    mixed_unlabel_dataset = DatasetFromList(P2, copy=False)
    OOD_unlabel_dataset = DatasetFromList(M2, copy=False)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    ID_label_dataset = MapDataset(ID_label_dataset, mapper)
    ID_unlabel_dataset = MapDataset(ID_unlabel_dataset, mapper)
    mixed_unlabel_dataset = MapDataset(mixed_unlabel_dataset, mapper)
    OOD_unlabel_dataset = MapDataset(OOD_unlabel_dataset, mapper)


    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        L_sampler = TrainingSampler(len(ID_label_dataset))
        U_sampler = TrainingSampler(len(ID_unlabel_dataset))
        M_sampler = TrainingSampler(len(mixed_unlabel_dataset))
        O_sampler = TrainingSampler(len(OOD_unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop(
        (ID_label_dataset, ID_unlabel_dataset, mixed_unlabel_dataset, OOD_unlabel_dataset),
        (L_sampler, U_sampler, M_sampler, O_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    L_dataset, U_dataset, M_dataset, O_dataset = dataset
    L_sampler, U_sampler, M_sampler, O_sampler = sampler

    if aspect_ratio_grouping:
        L_data_loader = torch.utils.data.DataLoader(
            L_dataset,
            sampler=L_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        U_data_loader = torch.utils.data.DataLoader(
            U_dataset,
            sampler=U_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        M_data_loader = torch.utils.data.DataLoader(
            M_dataset,
            sampler=M_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        O_data_loader = torch.utils.data.DataLoader(
            O_dataset,
            sampler=O_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (L_data_loader, U_data_loader, M_data_loader, O_data_loader),
            (batch_size_label, batch_size_unlabel, batch_size_unlabel, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")