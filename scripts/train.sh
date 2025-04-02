#!/bin/bash

python main.py --mode 'train' --input_channel 3 --output_channel 7 --dataset Skin \
    --epoch 50 --learning_rate 0.0064 --device 'cuda:0' --model 'DDS'

