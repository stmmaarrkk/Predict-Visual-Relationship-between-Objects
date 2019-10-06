from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from utils import array_tool as at
from .utils.bbox_tools import loc2bbox
from .utils.nms.non_maximum_suppression import non_maximum_suppression
from voc_dataset import Preprocess

from torch import nn
from torch.nn import functional as F
from ipdb import set_trace


class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scenario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(self, extractor, rpn, head_name, head_color,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head_name = head_name
        self.head_color = head_color

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')
        self.predict_pp = Preprocess(loc_normalize_mean, loc_normalize_std)

    @property
    def n_class(self, attr):
        # Total number of classes including the background.
        return self.head_name.n_class if attr == 'name' else self.head_name.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale)
        roi_name_locs, roi_name_scores = self.head_name(
            h, rois, roi_indices)
        roi_color_locs, roi_color_scores = self.head_color(
            h, rois, roi_indices)
        return roi_name_locs, roi_name_scores, roi_color_locs, roi_color_scores, rois, roi_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.7
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.6
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def choose_final_index(self, length, keep_name_, keep_color_, count_thresh=2):
        assert count_thresh in [1, 2]

        final_keep = [val for val in keep_name_ if val in keep_color_]
        """
        final_keep = list()
        count_list = np.zeros(length, dtype=int)
        for num_list in [keep_name_, keep_color_]:
            for num in num_list:
                count_list[num] += 1
                if count_list[num] == count_thresh:
                    final_keep.append(num)
        final_keep = np.asarray(final_keep)
        """
        return final_keep

    def _suppress(self, raw_name_bboxes_, raw_name_probs_, raw_color_bboxes_, raw_color_probs_, prob_thresh):
        # 本身class先or,再去跟shape和color 做or
        # print("in suppress: bbox:{0}, prob:{1}".format(raw_name_bboxes_.shape, raw_prob.shape))
        # NMS for final output
        bboxes_ = list()
        names_ = list()
        colors_ = list()
        name_scores_ = list()
        color_scores_ = list()
        # skip cls_id = 0 because it is the background class

        keep_name = [] # keep_name[0] is the list contain the most union of set
        keep_color = []

        cls_bbox_name = raw_name_bboxes_.reshape((-1, self.head_name.n_class, 4))
        cls_bbox_color = raw_color_bboxes_.reshape((-1, self.head_color.n_class, 4))

        for l in range(1, self.head_name.n_class):
            ##
            name_bboxes_l = raw_name_bboxes_.reshape((-1, self.head_name.n_class, 4))[:, l, :]  # certain class
            name_probs_l = raw_name_probs_[:, l]
            mask = name_probs_l >= prob_thresh
            name_bboxes_l = name_bboxes_l[mask]
            name_probs_l = name_probs_l[mask]
            keep = non_maximum_suppression(
                cp.array(name_bboxes_l), self.nms_thresh,
                name_probs_l)  # return an array of index sorted by score in decending order
            keep_name.append(cp.asnumpy(keep))
            keep_name[0] = np.unique(np.hstack((keep_name[0], keep_name[-1])))
        for l in range(1, self.head_color.n_class):
            ##
            color_bboxes_l = raw_color_bboxes_.reshape((-1, self.head_color.n_class, 4))[:, l, :]  # certain class
            color_probs_l = raw_color_probs_[:, l]
            mask = color_probs_l >= prob_thresh
            color_bboxes_l = color_bboxes_l[mask]
            color_probs_l = color_probs_l[mask]
            keep = non_maximum_suppression(
                cp.array(color_bboxes_l), self.nms_thresh,
                color_probs_l)  # return an array of index sorted by score in decending order
            keep_color.append(cp.asnumpy(keep))
            keep_color[0] = np.unique(np.hstack((keep_color[0], keep_color[-1])))

        ##choose the final_keep_idx
        final_keep = \
            self.choose_final_index(raw_name_bboxes_.shape[0], keep_name[0], keep_color[0], count_thresh=2)
        #set_trace()
        for index in final_keep:
            # decide labels
            # check the highest score in each attrobute
            highest_name_score_index = np.argmax(raw_name_probs_[index][1:])  # return 0~2
            highest_color_score_index = np.argmax(raw_color_probs_[index][1:])
            # The labels are in [0, self.n_class - 2], so -1
            names_.append(highest_name_score_index)
            colors_.append(highest_color_score_index)
            # scores of names
            highest_name_score = raw_name_probs_[index][highest_name_score_index + 1]
            highest_color_score = raw_color_probs_[index][highest_color_score_index + 1]
            name_scores_.append(highest_name_score)
            color_scores_.append(highest_color_score)
            # decide bbox
            highest_list = np.array([highest_name_score, highest_color_score])
            highest_among = np.argmax(highest_list)
            assert highest_among in [0, 1]
            if highest_among == 0:  # name has highest score
                bboxes_.append(cls_bbox_name[index][highest_name_score_index + 1])
                # print("0 box append {0}".format(cls_bbox[index][highest_name_score_index].shape))
            else:  # color has highest score
                bboxes_.append(cls_bbox_color[index][highest_color_score_index + 1])
                # print("1 box append {0}".format(cls_bbox_color[index][highest_color_score_index].shape))
        ##transform into numpy, concatenate all in one seq
        bboxes_ = np.asarray(bboxes_).astype(np.float32)
        names_ = np.asarray(names_).astype(np.int32)
        colors_ = np.asarray(colors_).astype(np.int32)
        name_scores_ = np.asarray(name_scores_).astype(np.float32)
        color_scores_ = np.asarray(color_scores_).astype(np.float32)
        """
        print("bbox shape is {0}".format(bbox.shape))
        print("names shape is {0}".format(names.shape))
        print("names_shape shape is {0}".format(label_shape.shape))
        print("colors shape is {0}".format(colors.shape))
        print("score shape is {0}".format(score.shape))
        print("score_shape shape is {0}".format(score_shape.shape))
        print("color_scores shape is {0}".format(color_scores.shape))
        """
        return bboxes_, names_, name_scores_, colors_, color_scores_

    def _pred_transform(self, img):  # For image only prediction
        w_o, h_o = img.size  # Original
        img = self.predict_pp.img_transform(img)  # (H,W) RGB -> torch.Tensor with size()=(C,H,W),scaled
        w_p, h_p = img.size(2), img.size(1)
        scale = h_p / h_o
        return img[None], t.tensor(scale)[None]  # add batch Dim

    def test_img(self, img, visualize=True):
        """

        Args:
            img: Image.Image

        Returns:

        """
        img, scale = self._pred_transform(img)
        bbox, label, score = self.predict(img, scale, visualize=visualize)
        return bbox, label, score

    def predict(self, img, scale, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            img Image object
        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
        else:
            self.use_preset('evaluate')
        with t.no_grad():
            # size = [img.size[1], img.size[0]]  # H,W
            # img = at.to_tensor(self.pred_transform(img)).float()[None]  # add batch dimension
            # scale = img.size(3) / size[1]  # w_new/w_old
            # # recover original size
            scale = np.asscalar(scale.numpy()[0])
            size = np.round([img.size(2) / scale, img.size(3) / scale])
            img = img.float().cuda()

            roi_name_locs, roi_name_scores, roi_color_locs, roi_color_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_name_scores = roi_name_scores.data
            roi_name_locs = roi_name_locs.data
            # color
            roi_color_scores = roi_color_scores.data
            roi_color_locs = roi_color_locs.data

            roi_org = at.to_tensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.head_name.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.head_name.n_class)[None]

            roi_name_locs = (roi_name_locs * std + mean)
            roi_name_locs = roi_name_locs.view(-1, self.head_name.n_class, 4)
            roi = roi_org.view(-1, 1, 4).expand_as(roi_name_locs)
            name_bbox = loc2bbox(at.to_np(roi).reshape((-1, 4)),
                                at.to_np(roi_name_locs).reshape((-1, 4)))
            name_bbox = at.to_tensor(name_bbox)
            name_bbox = name_bbox.view(-1, self.head_name.n_class * 4)
            # clip bounding box
            name_bbox[:, 0::2] = (name_bbox[:, 0::2]).clamp(min=0, max=size[0])
            name_bbox[:, 1::2] = (name_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.to_np(F.softmax(at.to_tensor(roi_name_scores), dim=1))
            # print("prob : {0}".format(prob.shape))#(300,4)
            raw_name_bboxes_ = at.to_np(name_bbox)
            raw_name_probs_ = at.to_np(prob)

            ##color
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.head_color.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.head_color.n_class)[None]

            roi_color_locs = (roi_color_locs * std + mean)
            roi_color_locs = roi_color_locs.view(-1, self.head_color.n_class, 4)
            roi = roi_org.view(-1, 1, 4).expand_as(roi_color_locs)
            color_bbox = loc2bbox(at.to_np(roi).reshape((-1, 4)),
                                 at.to_np(roi_color_locs).reshape((-1, 4)))
            color_bbox = at.to_tensor(color_bbox)
            color_bbox = color_bbox.view(-1, self.head_color.n_class * 4)
            # clip bounding box
            color_bbox[:, 0::2] = (color_bbox[:, 0::2]).clamp(min=0, max=size[0])
            color_bbox[:, 1::2] = (color_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.to_np(F.softmax(at.to_tensor(roi_color_scores), dim=1))
            # print("prob : {0}".format(prob.shape))#(300,4)
            raw_color_bboxes_ = at.to_np(color_bbox)
            raw_color_probs_ = at.to_np(prob)

            bboxes_, names_, name_scores_, colors_, color_scores_ \
                = self._suppress(raw_name_bboxes_, raw_name_probs_, raw_color_bboxes_, raw_color_probs_, prob_thresh=0.7)
            """
            print('name')
            print(names_, name_scores_)
            print('color')
            print(colors_, color_scores_)
            set_trace()
            """
            if raw_name_bboxes_.shape[0] != raw_color_bboxes_.shape[0]:
                print("raw_name_bboxes_ {0}".format(raw_name_bboxes_.shape))
                print("raw_color_bboxes_ {0}".format(raw_color_bboxes_.shape))
                a = 1
                assert a == 0, "raw is not equal"
            if names_.shape[0] != colors_.shape[0]:
                print("names_ {0}".format(names_.shape))
                print("colors_ {0}".format(colors_.shape))
                a = 1
                print("bboxes_ == ".format(bboxes_.shape))
                assert a == 0, "names_ is not equal"

        self.use_preset('evaluate')
        self.train()
        return bboxes_, names_, name_scores_, colors_, color_scores_

    # def get_optimizer(self):
    #     """
    #     return optimizer, It could be overwriten if you want to specify
    #     special optimizer
    #     """
    #     lr = opt.lr
    #     params = []
    #     for key, value in dict(self.named_parameters()).items():
    #         if value.requires_grad:
    #             if 'bias' in key:
    #                 params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
    #             else:
    #                 params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
    #     if opt.use_adam:
    #         self.optimizer = t.optim.Adam(params)
    #     else:
    #         self.optimizer = t.optim.SGD(params, momentum=0.9)
    #     return self.optimizer
    #
    # def scale_lr(self, decay=0.1):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] *= decay
    #     return self.optimizer
