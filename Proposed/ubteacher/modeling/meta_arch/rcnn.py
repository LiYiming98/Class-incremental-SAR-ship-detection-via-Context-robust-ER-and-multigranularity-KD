# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import deque

import numpy as np
import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import ImageList
from sklearn.metrics.pairwise import cosine_similarity
import random

CLASSES = ['Container', 'Fishing', 'cell-container', 'ore-oil', 'LawEnforce', 'Dredger']

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def normalize(self, image):
        """标准化处理"""
        dtype, device = image[0].dtype, image[0].device
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return [(img - mean[:, None, None]) / std[:, None, None] for img in image]

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - 0.0) / 255.0 for x in images]
        images = self.normalize(images)
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        return images

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, tea_proposals=None, iter=None, burn_up=None
    ):
        if (not self.training) and (not val_mode) and (not branch[-5:] == 'guide') and (not branch == 'mixup') and (not branch == 'prototype'):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)  # extra image features from backbone network
        if branch == 'prototype':
            box_feats = self.roi_heads._forward_box(features, given_proposals, branch=branch)
            return box_feats
        if branch == 'mixup':
            box_feats, ground_truth = self.roi_heads._forward_box(features, gt_instances, branch=branch)
            return box_feats, ground_truth
        if branch == 'teacher_guide':
            proposals_rpn, proposal_losses = self.proposal_generator(images, features, gt_instances)
            all_selected_proposals = []
            for k in range(len(proposals_rpn)):
                # sort proposals according to their objectness score (detectron2框架已默认实现)
                proposals = proposals_rpn[k]
                # # choose first 128 highest objectness score proposals  and then random choose 64 proposals from them
                # list = range(0, 128, 1)
                # selected_proposal_index = random.sample(list, 64)
                # choose the potential target proposals (objectness_logits > -3)
                selected_proposal_index = (proposals._fields['objectness_logits'] > -0.8473).nonzero(as_tuple=True)[0]
                # retain the random 64 proposals
                selected_proposal_score = proposals._fields['objectness_logits'][selected_proposal_index]
                selected_proposal_bbox = proposals._fields['proposal_boxes'][selected_proposal_index]
                proposals._fields['objectness_logits'] = selected_proposal_score
                proposals._fields['proposal_boxes'] = selected_proposal_bbox
                all_selected_proposals.append(proposals)
            # generate teacher proposal labels
            box_features, pred_instances, predictions = self.roi_heads._forward_box(features, proposals_rpn, branch='teacher_guide')# use ROI-subnet to generate final results
            return features, box_features, pred_instances, predictions, all_selected_proposals
        if branch == 'stu_supervised':
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # roi_head lower branch
            _, detector_losses = self.roi_heads(images, features, proposals_rpn, gt_instances, branch=branch)
            box_features, pred_instances, predictions = self.roi_heads._forward_box(features, tea_proposals, branch='teacher_guide')
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, features, [box_features, pred_instances, predictions], None

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, features, [], None
        elif branch == "unsup_data_weak":


            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
