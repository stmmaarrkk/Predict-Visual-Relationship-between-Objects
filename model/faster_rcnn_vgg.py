import torch as t
from torch import nn
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
from utils import array_tool as at
from .region_proposal_network import RegionProposalNetwork
from .faster_rcnn import FasterRCNN
from .roi_module import RoIPooling2D
from ipdb import set_trace


def decom_vgg(pretrained, version, freeze_idx, freeze_clf, use_drop=False):
    assert version in ['VGG_16', 'VGG_16_bn', 'VGG_19', 'VGG_19_bn']
    model = {
        'VGG_16': lambda pre: vgg16(pre),
        'VGG_16_bn': lambda pre: vgg16_bn(pre),
        'VGG_19': lambda pre: vgg19(pre),
        'VGG_19_bn': lambda pre: vgg19_bn(pre)
    }[version](pretrained)

    features = list(model.features)[:-1]  ##throw out Max pooling

    if freeze_idx == -1:  ##means freeze all layer
        freeze_idx = len(features)
    # freeze
    for layer in features[:freeze_idx]:
        for p in layer.parameters():
            p.requires_grad = False

    name_clf = list(model.classifier)[:-1]
    ##create color clf##
    model = {
        'VGG_16': lambda pre: vgg16(pre),
        'VGG_16_bn': lambda pre: vgg16_bn(pre),
        'VGG_19': lambda pre: vgg19(pre),
        'VGG_19_bn': lambda pre: vgg19_bn(pre)
    }[version](pretrained)
    color_clf = list(model.classifier)[:-1]

    if not use_drop:
        del name_clf[5]
        del color_clf[5]
        del name_clf[2]
        del color_clf[2]

    name_clf = nn.Sequential(*name_clf)
    color_clf = nn.Sequential(*color_clf)

    if freeze_clf:
        for name_clf_l, color_clf_l in zip(name_clf[:len(name_clf)], color_clf[:len(color_clf)]):
            for p in name_clf_l.parameters():
                p.requires_grad = False
            for p in color_clf_l.parameters():
                p.requires_grad = False

    return nn.Sequential(*features), name_clf, color_clf


class FasterRCNN_VGG(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self, version,
                 n_names,
                 n_colors,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 freeze_idx=-1,
                 freeze_clf=False,
                 vgg_pre=True,
                 use_drop=False):
        extractor, clf_name, clf_color = decom_vgg(vgg_pre, version=version, freeze_idx=freeze_idx,
                                                   freeze_clf=freeze_clf, use_drop=use_drop)

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head_name = VGG16RoIHead(
            n_class=n_names + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=clf_name
        )
        head_color = VGG16RoIHead(
            n_class=n_colors + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=clf_color
        )

        super(FasterRCNN_VGG, self).__init__(
            extractor,
            rpn,
            head_name,
            head_color,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.to_tensor(roi_indices).float()
        rois = at.to_tensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: the prediction of roi will be tensor type([N,H,W]), so convert back from yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
