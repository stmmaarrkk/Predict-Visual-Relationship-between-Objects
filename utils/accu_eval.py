import numpy as np
from model.utils.bbox_tools import bbox_iou
from PIL import Image
from plot_tool import display_image
from plot_tool import vis_bbox
from img_type import tensor_to_PIL
import matplotlib.pyplot as plt

def accu_eval(imgs, pred_bboxes, pred_labels, pred_scores, pred_labels_shape , pred_scores_shape, pred_labels_color ,
            gt_bboxes, gt_labels, gt_labels_shape, gt_labels_color, iou_thresh=0.5): #these arg are lists contain python
    print("accu_eval")
    ##predicted##
    """
    pred_bboxes = np.concatenate(pred_bboxes[:])
    pred_labels = np.array(pred_labels)
    pred_scores = np.array(pred_scores)
    pred_labels_shape = np.array(pred_labels_shape)
    pred_scores_shape = np.array(pred_scores_shape)
    pred_labels_color = np.array(pred_labels_color)
    pred_scores_color = np.array(pred_scores_color)
    print("pred_scores_color below")
    print(pred_bboxes.shape)

    for gt_bboxes_, gt_labels_, gt_labels_shape_, gt_labels_color_,  \
            in gt_bboxes, gt_labels, gt_labels_shape, gt_labels_color, pred_bboxes: #each means a img
    """
    print(len(gt_bboxes))
    print(len(gt_labels))
    print(len(gt_labels_shape))
    print(len(gt_labels_color))

    accuracy_mat = {"name": np.array([0, 0]), "shape": np.array([0, 0]), "color": np.array([0, 0])}  # right/ total
    for img_index, img in enumerate(imgs): #each loop means a img
        print("\n-------------Img [{0}]----------------".format(img_index))
        ##gt##
        gt_bboxes_ = gt_bboxes[img_index]
        gt_labels_ = gt_labels[img_index]
        gt_labels_shape_ = gt_labels_shape[img_index]
        gt_labels_color_ = gt_labels_color[img_index]
        ##predict##
        pred_bboxes_ = pred_bboxes[img_index]
        pred_labels_ = pred_labels[img_index]
        pred_scores_ = pred_scores[img_index]
        pred_labels_shape_ = pred_labels_shape[img_index]
        pred_scores_shape_ = pred_scores_shape[img_index]
        pred_labels_color_ = pred_labels_color[img_index]
        pred_scores_color_ = pred_scores_color[img_index]

        iou = bbox_iou(gt_bboxes_, pred_bboxes_)  # input is (N,4) (K,4) output is (N, K), all are numpy
        assert pred_bboxes_.shape[0]==pred_labels_.shape[0]
        assert pred_bboxes_.shape[0]==pred_scores_.shape[0]
        print("There are {0} predict bboxes".format(pred_bboxes_.shape[0]))
        for bbox_index in range(gt_bboxes_.shape[0]): #each loop means a gt_bbox in a img
            print("#######Bbox [{0}]########".format(bbox_index))
            gt_index = np.argmax(iou[bbox_index])
            if iou[bbox_index][gt_index] > iou_thresh:  #Big enough to print
                label = pred_labels_[gt_index]
                score = pred_scores_[gt_index]
                label_shape = pred_labels_shape_[gt_index]
                score_shape = pred_scores_shape_[gt_index]
                label_color = pred_labels_color_[gt_index]
                score_color = pred_scores_color_[gt_index]

                print("The IOU is: {0} %".format(iou[bbox_index][gt_index]*100))
                if label == gt_labels_[bbox_index]:
                    accuracy_mat["name"] += [1, 1]
                    print("label is correctly predicted as {0} with {1} %".format(label,score*100))
                else :
                    accuracy_mat["name"] += [0, 1]
                    print("label is mispredicted as {0} with {1} %, the answer is {2}".format(label, score*100,  gt_labels_[bbox_index]))
                if label_shape == gt_labels_shape_[bbox_index]:
                    accuracy_mat["shape"] += [1, 1]
                    print("shape is correctly predicted as {0} with {1} %".format(label_shape, score_shape*100))
                else :
                    accuracy_mat["shape"] += [0, 1]
                    print("shape is mispredicted as {0} with {1} %, the answer is {2}".format(label_shape, score_shape*100, gt_labels_shape_[bbox_index]))
                if label_color == gt_labels_color_[bbox_index]:
                    accuracy_mat["color"] += [1, 1]
                    print("color is correctly predicted as {0} with {1} %".format(label_color, score_color*100))
                else :
                    accuracy_mat["color"] += [0, 1]
                    print("color is mispredicted as {0} with {1} %, the answer is {2}".format(label_color, score_color*100, gt_labels_color_[bbox_index]))
            else:
                accuracy_mat["name"] += [0, 1]
                accuracy_mat["shape"] += [0, 1]
                accuracy_mat["color"] += [0, 1]
                print("No match for iou!!")
        del iou
        ##visualize
        vis_bbox(tensor_to_PIL(img), pred_bboxes_, pred_labels_, pred_scores_,
                 pred_labels_shape_, pred_scores_shape_, pred_labels_color_, pred_scores_color_)
        plt.show()
    print("LABEL: correct: {0}, total: {1}, accuracy: {2}"\
          .format(accuracy_mat["name"][0], accuracy_mat["name"][1],
                  accuracy_mat["name"][0]/accuracy_mat["name"][1]))
    print("SHAPE: correct: {0}, total: {1}, accuracy: {2}" \
          .format(accuracy_mat["shape"][0], accuracy_mat["shape"][1],
                  accuracy_mat["shape"][0] / accuracy_mat["shape"][1]))
    print("COLOR: correct: {0}, total: {1}, accuracy: {2}" \
          .format(accuracy_mat["color"][0], accuracy_mat["color"][1],
                  accuracy_mat["color"][0] / accuracy_mat["color"][1]))
