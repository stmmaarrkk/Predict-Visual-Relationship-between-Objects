import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
from ipdb import set_trace

NAMES = ['bowl', 'box', 'basket', 'loofah', 'can']
SHAPES = ['cube', 'container', 'cylinder']
COLORS = ['red', 'yellow', 'green', 'blue']
RELATION = ['no', 'on', 'in']
# PIL.Image.BICUBIC

def get_scale_transform(img_min, img_max):
    def scale_transform(img):
        W, H = img.size
        scale1 = img_min / min(H, W)
        scale2 = img_max / max(H, W)
        scale = min(scale1, scale2)
        img = img.resize((int(W * scale), int(H * scale)), Image.BICUBIC)
        return img

    return scale_transform


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


class Preprocess:
    def __init__(self, normalize_means, normalize_stds, img_min_size=600, img_max_size=1000, p_hflip=0.5,
                 bbox_resize=True):
        """
        Preprocess the data for VOCBoxDataset
        :param normalize_means: mean for torchvision.transforms.Normalize()
        :param normalize_stds:  std for torchvision.transforms.Normalize()
        :param img_min_size: minimum length of the shorter edge for the resized image
        :param img_max_size: maximum length of the longer edge for the resized image \
        :param p_hflip: Probability of doing random horizontal flip
        :param image_only: transform images, return scale and trasnsformed image, not from VOC dataset
        """
        self.scale_transform = get_scale_transform(img_min_size, img_max_size)
        self.img_transform = transforms.Compose([ ##transform into tensor type for training
            lambda img: self.scale_transform(img),
            lambda img: np.asarray(img),  # H,W,C
            transforms.ToTensor(),  # C,H,W with value range [0-1] float tensor is pil image format
            transforms.Normalize(normalize_means, normalize_stds),
        ])
        self.prob_hflip = p_hflip
        self.bbox_resize = bbox_resize

    def __call__(self, data): #to process the img
        img_name, img, index, bbox, name, shape, color, rel_mat = data  # img=Image Object(W,H),RGB , bbox=(N,4) label=N
        w_o, h_o = img.size  # Original
        assert w_o, h_o == (640,480)
        # Process Image
        flip = False
        if np.random.uniform(0, 1) < self.prob_hflip: ##flip the img
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip = True
        img = self.img_transform(img)  # (H,W) RGB -> torch.Tensor with size()=(C,H,W)
        # Process BBox
        w_p, h_p = img.size(2), img.size(1)  # Transformed
        assert w_p, h_p == (1000, 600)
        scale = h_p / h_o
        if self.bbox_resize:
            bbox = resize_bbox(bbox, (h_o, w_o), (h_p, w_p))
        if flip:
            bbox = flip_bbox(bbox, (h_p, w_p), x_flip=True) #to flip bbox

        #preprocessing relation
        rel_mat = rel_mat.replace('[', '')
        rel_mat = rel_mat.replace(']', '')
        rel_mat = rel_mat.replace('.', '')
        rel_mat = rel_mat.split(' ')
        rel_mat = [int(e) for e in rel_mat]
        relation = np.array(rel_mat).reshape((bbox.shape[0],-1)) #reshape to NxN

        return img_name, img, index, bbox, name, shape, color, relation, scale  # , return after processing

class RD_dataset(Dataset):
    def __init__(self, data_dir, split='train', preprocess=None):
        """

        :param str data_dir: dir to place ImageSets
        :param str split: 'trainval','test'
        """

        assert split in ['trainval', 'test']
        #assert not (dataset_name == '2012_part' and split == 'test'), 'No Annotation Data in VOC2012 Test set'

        id_list_file = os.path.join(
            data_dir, 'ImageSets/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

        self.preprocess = preprocess

        self.img_default_transform = transforms.Compose([
            lambda img: np.asarray(img),  # H,W,C
            transforms.ToTensor(),  # C,H,W with value range [0-1] float
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        """
        :param int i: the index of the image data in the dataset

        The bounding boxes are packed into a two dimensional tensor of shape
        :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
        the image. The second axis represents attributes of the bounding box.
        They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
        four attributes are coordinates of the top left and the bottom right
        vertices.

        The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
        :math:`R` is the number of bounding boxes in the image.
        The class name of the label :math:`l` is :math:`l` th element of
        :obj:`VOC_BBOX_LABEL_NAMES`.

        The type of the image, the bounding boxes and the labels are as follows.

        * :obj:`img.dtype == numpy.float32`
        * :obj:`bbox.dtype == numpy.float32`
        * :obj:`label.dtype == numpy.int32`
		* :obj:`shape.dtype == numpy.int32`
		* :obj:`color.dtype == numpy.int32`

        """
        id_ = self.ids[i]  # real image id
        #print(id_)
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml')) ##open the annotation file
        bbox = []  # in (ymin, xmin, ymax, xmax)
        name = []
        shape = []
        color = []
        index = []
        for idx, obj in enumerate(anno.findall('object')):
            ##find bbox
            bndbox_anno = obj.find('bndbox')
            temp = [0, 0, 0, 0]
            temp[0] = int(bndbox_anno.find('ymin').text)
            temp[1] = int(bndbox_anno.find('xmin').text)
            temp[2] = int(bndbox_anno.find('ymax').text)
            temp[3] = int(bndbox_anno.find('xmax').text)
            bbox.append(temp)

            ##find name
            name_anno = obj.find('name').text.lower().strip()
            assert name_anno in NAMES, "name:{0} in img:{1}".format(name_anno, id_)
            name.append(NAMES.index(name_anno))

            ##find shape
            shape_anno = obj.find('attributes').find('shape').text.lower().strip()
            assert shape_anno in SHAPES, "shape:{0} in img:{1}".format(shape_anno, id_)
            shape.append(SHAPES.index(shape_anno))

            ##find color
            color_anno = obj.find('attributes').find('color').text.lower().strip()
            assert color_anno in COLORS, "color:{0} in img:{1}".format(color_anno, id_)
            color.append(COLORS.index(color_anno))

            ##find index of the bbox##
            index.append(idx)
        ##process relation
        rel_mat = anno.find('relation').text
        bbox = np.stack(bbox).astype(np.float32)
        name = np.stack(name).astype(np.int32)
        shape = np.stack(shape).astype(np.int32)
        color = np.stack(color).astype(np.int32)
        index = np.stack(index).astype(np.int32)
        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.png')
        img = Image.open(img_file)  # H,W,3 => for using torchvision.transforms
        img_name = id_ + '.png'
        # Preprocess
        return self.preprocess((img_name, img, index, bbox, name, shape, color, rel_mat))

    def __len__(self):
        return len(self.ids)
