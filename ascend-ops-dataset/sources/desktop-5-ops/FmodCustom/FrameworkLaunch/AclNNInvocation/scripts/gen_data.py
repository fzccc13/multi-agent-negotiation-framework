#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import torch
import os

def gen_golden_data_simple():
    input_x = np.random.uniform(1, 100, [32]).astype(np.float16)
    input_y = np.random.uniform(1, 100, [32]).astype(np.float16)
    print("input_x.shape:",input_x.shape)

    tensor_x = torch.tensor(input_x)
    tensor_y = torch.tensor(input_y)
    golden = torch.fmod(tensor_x,tensor_y)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
