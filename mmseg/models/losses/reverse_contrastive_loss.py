import torch
import mmcv
import torch.nn as nn
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
import ipdb
from time import time
from ..builder import LOSSES
from .utils import weight_reduce_loss
from mmcv.utils import print_log
from mmseg.utils import get_root_logger


@LOSSES.register_module()
class ReverseContrastiveLoss(nn.Module):
    """KLPatchContrastiveLossBatch contained CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=0.1,
                 temp=10,
                 patch_size=100,
                 cal_function='COS07',
                 cal_gate=None,
                 loss_ratio=True,
                 down_sampling=False,
                 posweight=False):
        super(ReverseContrastiveLoss, self).__init__()
        if cal_gate is None:
            cal_gate = [0, 99]
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cal_size = patch_size
        self.cal_gate = cal_gate
        self.temp = temp
        self.loss_ratio = loss_ratio
        self.down_sampling = down_sampling
        self.posweight = posweight

        if cal_function == 'EQUAL':
            self.cons_func = self.calculate_equal_test
            self.uniform = False

    def cross_class(self, cls_score, label, b):
        # b, h, w = label.shape
        cls_result = torch.argmax(cls_score, dim=1) + 1  # [b,h,w]
        label = label + 1
        diff = torch.eq(cls_result - label, 0).int()  # [b,h,w] same:True,1; different:False,0
        # invert_diff = (cls_result - label).nonzero()  # bool can use ~
        invert_diff = 1 - diff
        b_ft_cross_region = [i for i in range(b)]
        b_tt_mask_region = b_ft_cross_region.copy()
        b_cross_class = b_ft_cross_region.copy()
        b_tt_class = b_ft_cross_region.copy()
        for i in range(b):
            ft_cross_region = []
            tt_mask_region = []
            tt_class = []
            cross_class = []
            # lists_labels_class: lists of labels' class
            lists_labels_class = torch.unique(label[i, :, :])  # optimize
            for j in lists_labels_class:
                j = j.item()
                if j == 256:
                    continue
                # tt: for class j, cls_result right classify,[h,w]
                tt = (cls_result[i, :, :] == j) * diff[i, :, :]  # bool
                # ft: for class j, cls_result false classify, may contain other class,[h,w]
                ft = (cls_result[i, :, :] == j) * invert_diff[i, :, :]  # bool
                cls_ft = label[i, :, :] * ft
                if len(tt_mask_region) == 0:
                    tt_mask_region = [tt]
                    tt_class = [j]
                else:
                    tt_mask_region.extend([tt])
                    tt_class.append(j)
                var = torch.unique(cls_ft)
                for k in var:
                    k = k.item()
                    if k == 256 or k == 255 or k == 0:
                        continue
                    if len(ft_cross_region) == 0:
                        ft_cross_region = [(label[i, :, :] == k) * ft]  # [h,w]
                        cross_class = [str(j) + '/' + str(k)]
                    else:
                        ft_cross_region.extend([(label[i, :, :] == k) * ft])
                        cross_class.append(str(j) + '/' + str(k))
            try:
                tt_mask_region = torch.stack(tt_mask_region, dim=-1)  # [c, h, w, TT_class]
            except:
                tt_mask_region = []
            try:
                ft_cross_region = torch.stack(ft_cross_region, dim=-1)  # [h,w,class]
            except:
                ft_cross_region = []
            b_ft_cross_region[i] = ft_cross_region
            b_cross_class[i] = cross_class
            b_tt_mask_region[i] = tt_mask_region
            b_tt_class[i] = tt_class

        return b_ft_cross_region, b_tt_mask_region, b_cross_class, b_tt_class

    def forward(self,
                cls_score,
                label,
                reduction_override=None,
                con_seg_logit=None,
                **kwargs):
        """Forward function belong cross region.

        Args:
            cls_score:[b,c,h,w]
            label:[b,h,w]
            reduction_override:
            con_seg_logit: [b,c,h/2,w/2]

        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        b, _, h, w = cls_score.shape
        _, c, h1, w1 = con_seg_logit.shape
        # ipdb.set_trace()
        gt_seg = F.interpolate(label.type(torch.cuda.FloatTensor), (h1, w1), mode='nearest').squeeze(1)
        cls_score = F.interpolate(cls_score, (h1, w1), mode='nearest')
        gt_seg = gt_seg.type(torch.cuda.LongTensor)
        b_ft_cross_region, b_tt_mask_region, b_cross_class, b_tt_class = \
            self.cross_class(cls_score, gt_seg, b)
        pos_weight = F.softmax(cls_score, 1).detach().max(1)[0]
        pos_weight = pos_weight.reshape(b, 1, h1 * w1)
        pos_weight_set = []
        pos_feature = []
        neg_feature = []
        cross_feature = []
        cross_length = 0
        con_seg_logit = con_seg_logit.reshape(b, c, h1 * w1)
        for i in range(b):
            cross_class = b_cross_class[i]
            for j, key in enumerate(cross_class):
                class_key1, class_key2 = key.split('/', 1)
                tt_index1 = b_tt_class[i].index(int(class_key1))
                tt_index2 = b_tt_class[i].index(int(class_key2))
                phi1 = torch.nonzero(b_tt_mask_region[i][:, :, tt_index1].reshape(h1 * w1))
                phi2 = torch.nonzero(b_tt_mask_region[i][:, :, tt_index2].reshape(h1 * w1))
                phi1_light = con_seg_logit[i, :, phi1]
                phi2_light = con_seg_logit[i, :, phi2]
                weight_light = pos_weight[i, :, phi2]
                cross_score = torch.nonzero(b_ft_cross_region[i][:, :, j].reshape(h1 * w1))
                cross_score_light = con_seg_logit[i, :, cross_score]

                phi1_length = phi1_light.shape[1]
                phi2_length = phi2_light.shape[1]
                cross_score_length = cross_score_light.shape[1]

                if phi2_length * cross_score_length == 0 \
                        or self.cal_gate[0] > cross_score_length \
                        or self.cal_gate[1] < cross_score_length:
                    continue
                reform_phi2_light = self.reform(phi2_light)
                reform_weight = self.reform(weight_light)
                reform_cross_light = self.reform(cross_score_light)
                '''
                In pascal context or ade20k, sometime negative label is not exist, so we 
                set negative feature same as cross feature
                '''
                if phi1_length == 0:
                    reform_phi1_light = reform_cross_light.clone()
                else:
                    reform_phi1_light = self.reform(phi1_light)
                cross_length += cross_score_length
                pos_feature.append(reform_phi2_light)
                pos_weight_set.append(reform_weight)
                neg_feature.append(reform_phi1_light)
                cross_feature.append(reform_cross_light)

        count = len(pos_feature)
        if count != 0:
            pos_feature = torch.cat(pos_feature, dim=2)
            pos_weight_set = torch.cat(pos_weight_set, dim=2)
            neg_feature = torch.cat(neg_feature, dim=2)
            cross_feature = torch.cat(cross_feature, dim=2)
            cross_feature = torch.transpose(cross_feature, 0, 2)
            neg_feature = torch.transpose(neg_feature, 0, 2)
            pos_feature = torch.transpose(pos_feature, 0, 2)
            pos_weight_set = torch.transpose(pos_weight_set, 0, 2)
            pos_weight_set = torch.transpose(pos_weight_set, 1, 2)
            logits = self.cons_func(cross_feature, neg_feature, pos_feature, count, cross_length, pos_weight_set)
            loss_cls = self.loss_weight * logits
            return loss_cls
        else:
            return (-cls_score.sum() + con_seg_logit.sum()) * 1e-16

    def reform(self, phi_light):
        step = self.cal_size // phi_light.shape[1]
        rest = self.cal_size % phi_light.shape[1]
        reform_features = []
        for i in range(step):
            reform_features.append(phi_light)
        reform_features.append(phi_light[:, :rest])
        reform_features = torch.cat(reform_features, dim=1)
        return reform_features

    def calculate_equal_test(self, cross_feature, neg_feature, pos_feature, count, cross_length, pos_weight_set):
        """Forward function belong equal cal.
        Args:
            pos_weight_set: [1, self.cal_size, count]
            cross_feature: [c, self.cal_size, count]
            pos_feature: [c, self.cal_size, count]
            neg_feature:[c, self.cal_size, count]
            count: nums of pairs
            cross_length: nums of calculate false classify pixel
        """
        cross_norm = torch.norm(cross_feature, p=2, dim=2).detach()
        pos_feature = pos_feature.detach()
        neg_feature = neg_feature.detach()

        pos_norm = torch.norm(pos_feature, p=2, dim=2)
        neg_norm = torch.norm(neg_feature, p=2, dim=2)
        # negtive sample pair

        # feature stack position change
        pos_dot_contrast = torch.div(torch.matmul(cross_feature,
                                                  torch.transpose(pos_feature, 1, 2), ),
                                     self.temp)
        neg_dot_contrast = torch.div(torch.matmul(cross_feature,
                                                  torch.transpose(neg_feature, 1, 2), ),
                                     self.temp)
        pos_scores = torch.exp(pos_dot_contrast /
                               cross_norm.unsqueeze(2) /
                               pos_norm.unsqueeze(1)) * pos_weight_set
        neg_scores = torch.exp(neg_dot_contrast /
                               cross_norm.unsqueeze(2) /
                               neg_norm.unsqueeze(1))

        logits = torch.log(neg_scores.sum(2) / pos_scores.sum(2) + 1).sum()
        logits = torch.div(logits, (count * self.cal_size)**2 / cross_length)

        return logits
