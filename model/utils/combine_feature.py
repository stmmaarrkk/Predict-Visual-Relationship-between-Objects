import numpy as np
import torch
import utils.array_tool as at
from model.roi_module import RoIPooling2D
import random as rd
from ipdb import set_trace
def get_combined_feature(feature, bboxes, use_spatial_feature=False, roi_size=7, spatial_scale=1 / 16 , flip=True):
    """
    :param feature: feature has passed extractor, shape[1,512,37,50]
    :param bboxes: shape[N, 4], N is num of object
    :return:
        combined_features: list of all combined feature(as same order as gt relation), is a list contains list of
        features in following type: [obj_a, obj_b, rel] or [obj_a, obj_b, rel, spatial_feature]
        rel_flip: list of rel that has been flipped
    """
    roi_pooling = RoIPooling2D(roi_size, roi_size, spatial_scale)  # spatial scale is not important
    num_of_bbox = bboxes.shape[0]
    #set_trace()
    ##bbox_scaling into fature scale
    ##TODO
    scale_x_imgtofeature, scale_y_imgtofeature = float(feature.size(3)/ 1000), float(feature.size(2) / 600)
    bboxes_f = np.zeros(bboxes.shape, dtype=int) ##bboxes_f is bboxes after resized to scale of feture map(37,50)
    for (bbox, bbox_f) in zip(bboxes, bboxes_f):
        bbox_f[0] = int(bbox[0] * scale_y_imgtofeature)
        bbox_f[2] = int(bbox[2] * scale_y_imgtofeature)
        bbox_f[1] = int(bbox[1] * scale_x_imgtofeature)
        bbox_f[3] = int(bbox[3] * scale_x_imgtofeature)
        assert bbox_f[0] in range(feature.size(2)+1) and bbox_f[2] in range(feature.size(2)+1) \
               and bbox_f[1] in range(feature.size(3)+1) and bbox_f[3] in range(feature.size(3)+1), "bbox:{0}  {1}".format(bbox, feature.shape)

    ##start forward
    rel_flip = []
    combined_features = []
    for i_obj_a in range(num_of_bbox):
        for i_obj_b in range(i_obj_a + 1, num_of_bbox):
            bbox_f_a = bboxes_f[i_obj_a]  # in (ymin, xmin, ymax, xmax)
            bbox_f_b = bboxes_f[i_obj_b]
            rel_flip.append(False)
            if rd.random() > 0.5 and flip:  # randomly swap obj1 and obj2
                bbox_f_a, bbox_f_b = bbox_f_b, bbox_f_a
                rel_flip[-1] = True
            ##get union cordinate
            union_cord = np.zeros(4)
            union_cord[0] = min(bbox_f_a[0], bbox_f_b[0])
            union_cord[1] = min(bbox_f_a[1], bbox_f_b[1])
            union_cord[2] = max(bbox_f_a[2], bbox_f_b[2])
            union_cord[3] = max(bbox_f_a[3], bbox_f_b[3])

            ##spatial feature
            if use_spatial_feature:
                ##dual spatial mask
                dual_channel_feature = torch.zeros(2, 37, 50)  # need modified
                for ii, bbox in enumerate([bbox_f_a, bbox_f_b]):  # obj_a_first
                    dual_channel_feature[ii, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
            ##combine features
            rois = np.stack([bbox_f_a, bbox_f_b, union_cord], axis=0)
            roi_indices = np.zeros(3)
            roi_indices = at.to_tensor(roi_indices).float()
            rois = at.to_tensor(rois).float()
            indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
            xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
            indices_and_rois = xy_indices_and_rois.contiguous()
            combined_feature_arr = roi_pooling(feature, indices_and_rois)  # in obj_a, obj_b, union oreder
            combined_feature = []
            for idx in range(combined_feature_arr.size(0)):
                combined_feature.append(torch.unsqueeze(combined_feature_arr[idx], 0)) #increase 1 dim
                assert combined_feature[-1].shape == (1, 512, 7, 7)
            combined_features.append(combined_feature)
    return combined_features, rel_flip