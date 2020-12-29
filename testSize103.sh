#!/bin/bash
set -eoux pipefail

python3 ./loadModelAndUse.py --train_size 100000 \
--val_size 5000 --image_pattern gaussian_rgb --demosaic_algo Malvar2004\
--image_size 103 --image_type both --crop none

# 127,129,131都可以使用.
