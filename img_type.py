import argparse
import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from torchvision.transforms import ToPILImage
from PIL import Image
class ToPILImage(object):
    """Convert a tensor to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving the value range.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.

        Returns:
            PIL.Image: Image converted to PIL.Image.

        """
        npimg = pic
        mode = None
        print("This is tensor")
        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()
        if torch.is_tensor(pic):
            print("going tensor")
        
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

            if npimg.dtype == np.uint8:
                mode = 'L'
                print("L RGB")
            if npimg.dtype == np.int16:
                mode = 'I;16'
                print("I;16 RGB")
            if npimg.dtype == np.int32:
                mode = 'I'
                print("I RGB")
            elif npimg.dtype == np.float32:
                mode = 'F'
                print("F RGB")
        else:
            if npimg.dtype == np.uint8:
                mode = 'RGB'
                print("going RGB")
        assert mode is not None, '{} is not supported'.format(npimg.dtype)
        return Image.fromarray(npimg, mode)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = ToPILImage()
    image = unloader(image)
    
    return image