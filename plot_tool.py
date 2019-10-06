import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torchvision.transforms as transforms

NAMES = ['bowl', 'box', 'basket', 'loofah', 'can']
SHAPES = ['cube', 'container', 'cylinder']
COLORS = ['red', 'yellow', 'green', 'blue']

def bbox_scaling(bboxes,org_size='480x640'): # to 600x800 apply to 480x640
    x_scale = 0
    y_scale = 0
    if org_size == '768x1024':
        x_scale = 0.78125
        y_scale = 0.78125
    else:
        x_scale = 1.25
        y_scale = 1.25
    bboxes = bboxes.copy()
    for bbox in bboxes:
        bbox[0] = int(y_scale * bbox[0])
        bbox[2] = int(y_scale * bbox[2])
        bbox[1] = int(x_scale * bbox[1])
        bbox[3] = int(x_scale * bbox[3])
    return bboxes


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torchvision.transforms as transforms
from ipdb import set_trace


def display_image(img):
    if isinstance(img, Image.Image):
        img.show()
        #plt.imshow(img)
        #plt.show()
    elif isinstance(img, np.ndarray):
        if img.shape[0] == 3:  # C,H,W
            img = img.transpose((1, 2, 0))  # H,W,C
        if np.max(img) > 1:  # 0-255 representation
            img = img.astype(np.uint8)
        img.show()
        #plt.imshow(img)
        #plt.show()

    elif isinstance(img, str):  # Path
        img = mpimg.imread(img)  # H,W,C
        img.show()
        #plt.imshow(img)
        #plt.show()
    else:
        raise NotImplementedError(
            'Unsupported Input Argument, expect Image, numpy array(0-255,0-1), or a path to an image file')


def vis_image(img, ax=None):
    img = denormalize_image(img)
    if isinstance(img, Image.Image):
        img = np.asarray(img).transpose((2, 0, 1))
    from matplotlib import pyplot as plot
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        # CHW -> HWC
        img = img.transpose((1, 2, 0))
        ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, gt_bbox, gt_name=None, gt_color=None,
             bbox=None, name=None, name_score=None, color=None, color_score=None,
             instance_colors=None, alpha=1., linewidth=3., ax=None):
    ax = vis_image(img, ax=ax)

    if isinstance(img, Image.Image):
        img = np.asarray(img).transpose((2, 0, 1))
    """
    To implememt

    img format must change from tensor to PIL
    """

    #combine gt and pred
    bbox = bbox_scaling(bbox)
    gt_bbox = bbox_scaling(gt_bbox)
    total_bbox = np.concatenate((gt_bbox,bbox), axis=0)

    #set_trace()


    ##color of bboxi
    instance_colors = np.zeros((total_bbox.shape[0], 3), dtype=np.float32)
    instance_colors[:gt_bbox.shape[0], 0] = 255 #gt_name is red
    instance_colors[gt_bbox.shape[0]:, 2] = 255  # pred_bbox is blue

    for i, bb in enumerate(total_bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]

        draw_color = instance_colors[i, :] / 255
        ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=draw_color, linewidth=linewidth, alpha=alpha))
        caption = []

        if i < gt_bbox.shape[0]: #show gt name
            caption.append('{0}\n'.format(NAMES[gt_name[i]]))
            caption.append('{0}\n'.format(COLORS[gt_color[i]]))
        else:  # show name and score
            score_index = i - gt_bbox.shape[0]
            caption.append(NAMES[name[score_index]])
            caption.append(':{:.2f}\n'.format(name_score[score_index]))
            caption.append(COLORS[color[score_index]])
            caption.append(':{:.2f}\n'.format(color_score[score_index]))

        if len(caption) > 0:
            ax.text(bb[1], bb[0], ''.join(caption), style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    return ax


def voc_colormap(labels):
    """
    If the color of the same label needs to be identical, call this

    """
    colors = []
    for label in labels:
        r, g, b = 0, 0, 0
        i = label
        for j in range(8):
            if i & (1 << 0):
                r |= 1 << (7 - j)
            if i & (1 << 1):
                g |= 1 << (7 - j)
            if i & (1 << 2):
                b |= 1 << (7 - j)
            i >>= 3
        colors.append((r, g, b))
    return np.array(colors, dtype=np.float32)


def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],):
    assert img.shape[0] == 3, 'Image must be in C,H,W format, but the shape of img is {0}'.format(img.shape)
    inv_normalize = transforms.Normalize(
        mean=np.divide(-np.array(mean), np.array(std)),
        std=1 / np.array(std)
    )
    img = inv_normalize(img)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    return (img * 255).astype(np.uint8)  # C,H,W 0-255 uint8 format

def print_tensor_photo(img):
    ax = vis_image(img, ax=None)