from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from .util import count_parameters as count
from .util import convert2cpu as cpu
from PIL import Image, ImageDraw
from scipy import ndimage

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    
    return canvas

        
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # (h, w, (bgr)) -> ((rgb), h, w)
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def letterbox_batch(batch, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    orig_w, orig_h = batch.shape[2], batch.shape[1]
    batch_size = batch.shape[0]
    w, h = inp_dim
    scale_w = w / orig_w
    scale_h = h / orig_h
    # new_w = int(orig_w * min(w / orig_w, h / orig_h))
    # new_h = int(orig_h * min(w / orig_w, h / orig_h))

    resized_batch = ndimage.zoom(batch, (1, scale_h, scale_w, 1))
    new_w, new_h = resized_batch.shape[2], resized_batch.shape[1]
    canvas_batch = np.full((batch_size, w, h, 3), 128)
    canvas_batch[:, (h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_batch

    return canvas_batch


def prep_image_batch(batch, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_batch = batch
    batch_size = orig_batch.shape[0]
    dim = orig_batch.shape[2], orig_batch.shape[1]
    print('start letterbox')
    batch = letterbox_batch(orig_batch, (inp_dim, inp_dim))
    batch_ = batch[:, :, :, ::-1].transpose((0, 3, 1, 2)).copy()   # (batch, h, w, (bgr)) -> (batch, (rgb), h, w)
    batch_ = torch.from_numpy(batch_).float().div(255.0)
    return batch_, orig_batch, dim


def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*network_dim, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3,*network_dim)
    img = img.float().div(255.0)
    return (img, orig_im, dim)

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]
    return inp


