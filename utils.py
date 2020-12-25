from typing import Any
import global_setting
import numpy as np
import cv2
import os
from colour_demosaicing import mosaicing_CFA_Bayer
from random import choice
from pathlib import Path


def demosaic(mosaiced, pattern='RGGB', algo='Malvar2004'):
    """
    We will show the original image, then the filtered image, 
    then the results of three different demosaicing algorithms.
    """
    demosaic_func = global_setting.demosaic_func_dict[algo]
    demosaiced = demosaic_func(mosaiced)
    return demosaiced

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out

def mosaic(im, pattern='RGGB'):
    """
    im should have rgb as 012 channels. And have shapes as (height, width, 3)
    Returns an image of the bayer filter pattern.
    """
    mosaiced = mosaicing_CFA_Bayer(im, pattern=pattern)
    return mosaiced


def rand_rgb_image(image_size, image_pattern):
    """
    Return image has RGB channels.
    """
    if image_pattern == 'gaussian_rgb':
        red = np.random.normal(0.6, 0.3, (image_size, image_size, 1))
        blue = np.random.normal(0.5, 0.25, (image_size, image_size, 1))
        green = np.random.normal(0.9, 0.4, (image_size, image_size, 1))
        image = np.concatenate((red, green, blue), axis=2)
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        image = (255 * image).astype('uint8')
    else:
        raise NotImplementedError()
    return image


def get_random_graph_from_a_path(path: str):
    #path = "P:\Pics02086910" # 125
    #path = "P:\mtfl\AFLW" # 127
    #path = "P:\mtfl\lfw_5590" # 129 & 127 & 131
    pathObject: Path = Path(path)
    random_path: str = str(choice(list(pathObject.iterdir())).resolve())
    image = cv2.imread(random_path)
    #image = cv2.resize(image,(150,150))
    return image


def jpeg_image(img, encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), 25]):
    processing_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, processing_img = cv2.imencode('.jpg', processing_img, encode_param)
    processing_img = cv2.imdecode(processing_img, 1)
    return cv2.cvtColor(processing_img, cv2.COLOR_BGR2RGB)


def debayer_image(img, bayer_pattern='RGGB', demosaic_algo='Malvar2004'):
    image_demosaiced = demosaic(
        mosaic(img, pattern=bayer_pattern),
        pattern=bayer_pattern,
        algo=demosaic_algo
    ).astype('uint8')
    return image_demosaiced
