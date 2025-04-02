#!/bin/bash

# options for eval
# --pretrained ---- Set path of the ckpt model.
# --eval ---- Set evaluation methods.
# --test_image ---- Set test image for gcam

python main.py --mode 'eval' --input_channel 3 --output_channel 7 --dataset Skin \
    --device 'cuda:0' --model 'DDS'

