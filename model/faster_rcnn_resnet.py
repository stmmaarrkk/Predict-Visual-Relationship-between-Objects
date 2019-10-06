import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn
from .region_proposal_network import RegionProposalNetwork
from .roi_module import RoIPooling2D
from utils import array_tool as at
from .faster_rcnn import FasterRCNN
import copy

def decom_resnet(pretrained, version, freeze_idx, freeze_clf, use_drop=False):
    assert version in ['ResNet_18','ResNet_34','ResNet_50','ResNet_101', 'ResNet_152']
    model = {
        'ResNet_18': lambda pre: resnet18(pre),
        'ResNet_34': lambda pre: resnet34(pre),
        'ResNet_50': lambda pre: resnet50(pre),
        'ResNet_101': lambda pre: resnet101(pre),
        'ResNet_152': lambda pre: resnet152(pre)
    }[version](pretrained)
    extractor = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                          model.layer1, model.layer2, model.layer3)

    if freeze_idx == -1:
        freeze_idx = 7

    # Fix blocks
    for idx in range(freeze_idx):
        for p in extractor[idx].parameters(): p.requires_grad = False
    """
    clf_deep = 256 if version in ['ResNet_18','ResNet_34'] else 1024
    name_clf = nn.Sequential(
        nn.Linear(clf_deep * 7 * 7, 2048),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(2048, 2048),
        nn.ReLU(True),
        nn.Dropout()
    )
    color_clf = nn.Sequential(
        nn.Linear(clf_deep * 7 * 7, 2048),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(2048, 2048),
        nn.ReLU(True),
        nn.Dropout()
    )

    if not use_drop:
        del name_clf[5]
        del color_clf[5]
        del name_clf[2]
        del color_clf[2]

    if freeze_clf:
        for name_clf_l, color_clf_l in zip(name_clf[:len(name_clf)], color_clf[:len(color_clf)]):
            for p in name_clf_l.parameters():
                p.requires_grad = False
            for p in color_clf_l.parameters():
                p.requires_grad = False
    """
    name_clf = nn.Sequential(model.layer4)
    color_clf = copy.copy(name_clf)
    return extractor, name_clf, color_clf

class FasterRCNN_ResNet(FasterRCNN):
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self, version='resnet_101',
                 n_names=5, n_colors=4,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 freeze_idx=-1,
                 freeze_clf=False,
                 pretrained=False,
                 use_drop = False):
        extractor, clf_name, clf_color = decom_resnet(pretrained, freeze_idx=freeze_idx, version=version,
                                                      freeze_clf=freeze_clf, use_drop= use_drop)

        if version in ['ResNet_18', 'ResNet_34']:
            rpn = RegionProposalNetwork(
                256, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
            )

            head_name = ResNetRoIHead(
                n_class=n_names + 1,
                roi_size=7,
                transfer_size=4096,
                spatial_scale=(1. / self.feat_stride),
                classifier=clf_name
            )

            head_color = ResNetRoIHead(
                n_class=n_colors + 1,
                roi_size=7,
                transfer_size=4096,
                spatial_scale=(1. / self.feat_stride),
                classifier=clf_color
            )
        else:
            rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
            )

            head_name = ResNetRoIHead(
                n_class=n_names + 1,
                roi_size=7,
                transfer_size=2048,
                spatial_scale=(1. / self.feat_stride),
                classifier=clf_name
            )

            head_color = ResNetRoIHead(
                n_class=n_colors + 1,
                roi_size=7,
                transfer_size=2048,
                spatial_scale=(1. / self.feat_stride),
                classifier=clf_color
            )

        super(FasterRCNN_ResNet, self).__init__(
            extractor,
            rpn,
            head_name,
            head_color,
        )

class ResNetRoIHead(nn.Module):
  def __init__(self, n_class, roi_size, transfer_size, spatial_scale ,classifier):
    # n_class includes the background
    super(ResNetRoIHead, self).__init__()
    self.classifier = classifier
    self.cls_loc = nn.Linear(transfer_size, 4 * n_class)
    self.score = nn.Linear(transfer_size, n_class)

    normal_init(self.cls_loc, 0, 0.001)
    normal_init(self.score, 0, 0.01)

    self.n_class = n_class
    self.roi_size = roi_size
    self.spatial_scale = spatial_scale
    self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale) #resnet
    self.downsample = nn.MaxPool3d((2,1,1), stride=(2,1,1))
  def forward(self, x, rois, roi_indices, if_color=False):
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
    # in case roi_indices is ndarray
    roi_indices = at.to_tensor(roi_indices).float()
    rois = at.to_tensor(rois).float()
    indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
    # NOTE: important: yx->xy
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    indices_and_rois = xy_indices_and_rois.contiguous()

    #print("shape of x: {0}".format(x.shape))
    #pool = self.roi_pool(x, indices_and_rois.view(-1, 5))
    #print("shape of pool: {0}".format(pool.shape)) #128,1024,7,7
    #print(pool.shape)
    pool = self.roi(x, indices_and_rois)
    #pool = pool.view(pool.size(0), -1) #hide when use layer 4 clf
    fc7 = self.classifier(pool).mean(3).mean(2) # [128, 7*7*512] #[128,7*7*1024] mean used when use layer 4 clf
    #print('fc7.shape',fc7.shape)
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