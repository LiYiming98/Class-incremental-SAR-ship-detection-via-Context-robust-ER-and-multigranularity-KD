import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_at(f_map, temp=2):
    N, C, H, W = f_map.shape

    value = torch.abs(f_map)
    # Bs*W*H
    fea_map = value.pow(temp).mean(axis=1, keepdim=True)
    # S_attention = (H * W * F.softmax(fea_map.view(N, -1)/temp, dim=1)).view(N, H, W)
    S_attention = (H * W * F.softmax(fea_map.view(N, -1), dim=1)).view(N, H, W)

    return S_attention


def activation_at_space(f_map, temp=2):
    N, C, H, W = f_map.shape

    value = torch.abs(f_map)
    # Bs*W*H
    fea_map = value.pow(temp).mean(axis=1, keepdim=True)
    # S_attention = (H * W * F.softmax(fea_map.view(N, -1)/temp, dim=1)).view(N, H, W)
    S_attention = (H * W * F.softmax(fea_map.view(N, -1), dim=1)).view(N, H, W)

    return S_attention


def activation_at_channel(f_map, temp=2):
    N, C, H, W = f_map.shape
    value = torch.abs(f_map)
    # 生成C*1*1的向量
    global_avg_pool = F.adaptive_avg_pool2d(value, (1, 1))
    C_attention = (C * F.softmax(global_avg_pool.view(N, -1), dim=1)).view(N, C)
    return C_attention


def afd_loss(f_map_s, f_map_t, S_t):
    loss_mse = nn.MSELoss(reduction='mean')
    S_t = S_t.unsqueeze(dim=1)

    fea_t = torch.mul(f_map_t, torch.sqrt(S_t))
    fea_s = torch.mul(f_map_s, torch.sqrt(S_t))

    loss = loss_mse(fea_s, fea_t)
    return loss


def fsc_loss(f_map_s, f_map_t, S_t, C_t):
    loss_mse = nn.MSELoss(reduction='mean')
    S_t = S_t.unsqueeze(dim=1)
    C_t = C_t.view(C_t.size(0), C_t.size(1), 1, 1)

    fea_t_s = torch.mul(f_map_t, torch.sqrt(S_t))
    fea_s_s = torch.mul(f_map_s, torch.sqrt(S_t))

    fea_t_c = torch.mul(f_map_t, torch.sqrt(C_t))
    fea_s_c = torch.mul(f_map_s, torch.sqrt(C_t))

    loss = loss_mse(fea_t_s, fea_s_s) + 0.1 * loss_mse(fea_t_c, fea_s_c)
    return loss


def pad_loss(S_s, S_t):
    loss_l1 = nn.L1Loss(reduction='mean')
    loss = loss_l1(S_s, S_t)
    return loss


def create_mask(image, boxes):
    _, height, width = image['image'].shape
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        mask[0, ymin:ymax, xmin:xmax] = 1
    return mask


def create_expanded_mask(image, boxes, scale=1.1):
    _, img_height, img_width = image['image'].shape
    mask = torch.zeros((1, img_height, img_width), dtype=torch.float32)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        center_x = (xmin + xmax) / 2
        center_y = (ymin +ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        # calculate new width and height
        new_width = width * (scale ** 0.5)
        new_height = height * (scale ** 0.5)
        # calculate new xmin, ymin, xmax, ymax
        new_xmin = center_x - new_width / 2
        new_ymin = center_y - new_height / 2
        new_xmax = center_x + new_width / 2
        new_ymax = center_y + new_height / 2

        # check if the new boundign box is within nthe image boundaries
        if (new_xmin >= 0 and new_ymin >= 0 and new_ymax <= img_height and new_xmax <= img_width):
            xmin, ymin, xmax, ymax = new_xmin, new_ymin, new_xmax, new_ymax
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        mask[0, ymin:ymax, xmin:xmax] = 1
    return mask


def scene_level_distillation_loss1(tea_feats, stu_feats, device='cuda'):
    """ 整张feature map特征图进行L1蒸馏 """
    loss = 0.0
    l1_loss = nn.L1Loss().to(device)

    for key in tea_feats.keys():
        if key in stu_feats:
            tea_feat = tea_feats[key].to(device)
            stu_feat = stu_feats[key].to(device)

            current_loss = l1_loss(stu_feat, tea_feat)
            loss += current_loss

    return loss


def scene_level_distillation_loss2(tea_feats, stu_feats, data, device='cuda'):
    """ 整张feature map特征图添加目标RoI掩膜，再进行L1蒸馏 """
    loss = 0.0
    l1_loss = nn.L1Loss().to(device)
    """ gt_boxes存储真实目标框，gt_classes存储真实目标类别 """
    """ 目前，输入的data中新旧目标均是完备标注的 """
    # 根据真实目标框构建相应的masks矩阵
    masks = []
    for item in data:
        gt_box = item['instances']._fields.get('gt_boxes')
        mask = create_mask(item, gt_box.tensor)
        masks.append(mask.repeat(256, 1, 1))
    masks = torch.stack(masks, dim=0)
    # 进行掩膜蒸馏
    for key in tea_feats.keys():
        if key in stu_feats:
            tea_feat = tea_feats[key].to(device)
            stu_feat = stu_feats[key].to(device)
            downsampled_masks = F.interpolate(masks, size=(tea_feat.size(2), tea_feat.size(3)), mode='bilinear').to(device)
            current_loss = l1_loss(stu_feat * downsampled_masks, tea_feat * downsampled_masks)
            loss += current_loss
    return loss


def scene_level_distillation_loss(tea_feats, stu_feats, data, device='cuda'):
    """
    Args:
        tea_feats(Tensor): Bs*C*H*W, teacher's feature map
        stu_feats(Tensor): Bs*C*H*W, student's feature map
        data(List):Bs
    """
    temp = 2
    """ 整张feature map特征图添加目标RoI掩膜，并进行上下文信息融合，再进行L1蒸馏 """
    loss = 0.0
    loss_mse = nn.MSELoss(reduction='mean')
    """ gt_boxes存储真实目标框，gt_classes存储真实目标类别 """
    """ 目前，输入的data中新旧目标均是完备标注的 """
    # 根据真实目标框构建相应的masks矩阵
    masks = []
    for item in data:
        gt_box = item['instances']._fields.get('gt_boxes')
        mask = create_expanded_mask(item, gt_box.tensor, scale=1.2)
        masks.append(mask.repeat(256, 1, 1))
    masks = torch.stack(masks, dim=0)
    # 进行掩膜蒸馏
    for key in tea_feats.keys():
        if key in stu_feats:
            tea_feat = tea_feats[key].to(device)
            stu_feat = stu_feats[key].to(device)
            S_attention_t = activation_at(tea_feat, temp)
            S_attention_s = activation_at(stu_feat, temp)

            loss_pad = pad_loss(S_attention_s, S_attention_t)
            downsampled_masks = F.interpolate(masks, size=(tea_feat.size(2), tea_feat.size(3)), mode='bilinear').to(device)
            loss_mask = loss_mse(stu_feat * downsampled_masks, tea_feat * downsampled_masks)
            loss += loss_pad + loss_mask
    return loss


def calculate_attentive_roi_feature_distillation(tea_RoI, stu_RoI, device='cuda'):
    """
    ICCV 2023 Augmented Boxes中的Attentive RoI Distillation (ARD) loss
    Args:
        tea_feats(Tensor): Bs*C*H*W, teacher's feature map
        stu_feats(Tensor): Bs*C*H*W, student's feature map
        data(List):Bs
    """
    """ 对RoI特征进行蒸馏 """
    S_attention_t = activation_at(tea_RoI, 2).to(device)
    S_attention_s = activation_at(stu_RoI, 2).to(device)
    loss_pad = pad_loss(S_attention_s, S_attention_t)
    loss_afd = afd_loss(tea_RoI, stu_RoI, S_attention_t)

    return loss_pad + loss_afd


def instance_level_distillation_loss(tea_RoI, stu_RoI, device='cuda'):
    """ 对RoI特征进行蒸馏 """
    # 空间注意力 return N*H*W
    S_attention_t = activation_at_space(tea_RoI, 2).to(device)
    S_attention_s = activation_at_space(stu_RoI, 2).to(device)
    # 通道注意力 return N*C
    C_attention_t = activation_at_channel(tea_RoI, 2).to(device)
    C_attention_s = activation_at_channel(stu_RoI, 2).to(device)

    loss_pad = pad_loss(S_attention_t, S_attention_s) + 0.1 * pad_loss(C_attention_t, C_attention_s)
    loss_afd = fsc_loss(tea_RoI, stu_RoI, S_attention_t, C_attention_t)
    #
    # loss_pad = pad_loss(S_attention_t, S_attention_s)
    # loss_afd = afd_loss(tea_RoI, stu_RoI, S_attention_t)

    return loss_afd + loss_pad


def logits_level_distillation_loss(tea_cls_logits, stu_cls_logits, tea_reg_logits, stu_reg_logits, device='cuda'):
    """
    Args:
        tea_cls_logits(Tensor):N*7
        stu_cls_logits(Tensor):N*7
        tea_reg_logits(Tensor):N*24
        stu_reg_logits(Tensor):N*24
    """
    """ 对顶层输出进行L1蒸馏 """
    num_of_distillation_categories = 4  # 1 + K
    num_of_distillation_categories_wo_bkg = 3  # K
    tot_classes_wo_bkg = stu_cls_logits.size()[1] - 1  # K+C
    # compute distillation loss

    # align the probobilities of the teacher model for the background class
    # with the probobilities of the student model for both background class([6]) and current classes(num_of_distillation_categories)

    new_bkg_idx = torch.tensor([x for x in range(num_of_distillation_categories_wo_bkg, tot_classes_wo_bkg)] + [6]).to(stu_cls_logits.device)
    den = torch.logsumexp(stu_cls_logits, dim=1)
    outputs_bkg = torch.logsumexp(torch.index_select(stu_cls_logits, index=new_bkg_idx, dim=1), dim=1) - den
    outputs_no_bgk = stu_cls_logits[:, 0:num_of_distillation_categories_wo_bkg] - den.unsqueeze(dim=1)
    # 然后，将教师模型输出分类向量也统一到1+K
    tea_den = torch.logsumexp(tea_cls_logits, dim=1)
    tea_outputs_bkg = torch.logsumexp(torch.index_select(tea_cls_logits, index=new_bkg_idx, dim=1), dim=1) - tea_den
    tea_outputs_no_bgk = tea_cls_logits[:, 0:num_of_distillation_categories_wo_bkg] - tea_den.unsqueeze(dim=1)
    tea_outputs_bkg = tea_outputs_bkg.unsqueeze(1)
    tea_outputs_new_cls = torch.cat((tea_outputs_no_bgk, tea_outputs_bkg), dim=1)
    labels = torch.softmax(tea_outputs_new_cls, dim=1)
    cls_distillation_loss = -torch.mean(((labels[:, :-1] * outputs_no_bgk).sum(dim=1) + labels[:, -1] * outputs_bkg) / num_of_distillation_categories)

    # 回归蒸馏损失
    l2_loss = nn.MSELoss(size_average=False, reduce=False)
    stu_reg_logits = stu_reg_logits[:, :4 * (num_of_distillation_categories - 1)]
    tea_reg_logits = tea_reg_logits[:, :4 * (num_of_distillation_categories - 1)]
    bbox_distillation_loss = l2_loss(stu_reg_logits, tea_reg_logits)
    bbox_distillation_loss = torch.mean(torch.mean(bbox_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
    instance_distillation_losses = torch.add(cls_distillation_loss, bbox_distillation_loss)
    return instance_distillation_losses

