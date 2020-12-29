#!/usr/bin/env python3
# coding=utf-8
# Training an infinitely large dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import global_setting
import os
import copy
import sys
from tqdm import tqdm
from config import get_config
from model_factory import get_model
from datasets_factory import get_dataloaders
from tools import get_optimizer, get_scheduler
from tools import get_dir_name, get_log_name
import cv2
from PIL import Image
from utils import rand_rgb_image, mosaic, demosaic, jpeg_image, debayer_image, get_random_graph_from_a_path, \
    gasuss_noise
import random

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

args, _ = get_config()

print(f"Using random seed {args.random_seed} to generate the images")
np.random.seed(args.random_seed)

log_name = get_log_name(args)
log_dir = get_dir_name(args.out_dir,
                       args.image_pattern,
                       args.image_type,
                       args.image_size,
                       args.demosaic_algo,
                       args.bayer_pattern,
                       args.crop,
                       args.crop_size,
                       log_name)
if os.path.exists(log_dir):
    print(f"Log dir {log_dir} already exists. It will be overwritten.")
else:
    os.makedirs(log_dir)
print(args)
model_save_path = os.path.join(log_dir, "model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_save_path, map_location=torch.device(device))
# model = get_model(args.model_architecture).to(args.device)
# model.load_state_dict(torch.load(model_save_path))
model.eval()


# 127,129,131都是人脸识别模型.
# 103为随机图片模型.
# TODO, 预测图片加噪音观察现象.
def pridict(randomValue: int):
    # 获取测试图片，并行相应的处理
    img = jpeg_image(debayer_image(rand_rgb_image(103, "gaussian_rgb")))
    img = get_random_graph_from_a_path()
    #   img = gasuss_noise(img, 0, 0.01)
    cv2.imshow("oirigin", img)
    if randomValue % 2 == 0:
        cv2.flip(img, 1, img)
    cv2.imshow("cv2_flip", img)
    cv2.waitKey()
    # img = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape)
    img = img.to(device)
    # print(img.shape)
    # print(device)
    with torch.no_grad():
        py = model(img)
    _, predicted = torch.max(py, 1)  # 获取分类结果
    classIndex_ = predicted[0]
    print(_, predicted)
    print("输入:{}".format(randomValue))
    print('预期结果', randomValue % 2 == 1)
    print('预测结果', classIndex_.item() == 1)
    try:
        assert ((randomValue % 2 == 1) == (classIndex_.item() == 1))
    except AssertionError:
        return 1
    return 0


if __name__ == '__main__':
    errorNumber: int = 0
    print(model_save_path)
    for i in range(0, 100):
        errorNumber += pridict(i)
    print(errorNumber)
