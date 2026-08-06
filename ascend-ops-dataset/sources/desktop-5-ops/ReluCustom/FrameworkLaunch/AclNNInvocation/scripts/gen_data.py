import torch
import torch.nn as nn
import numpy as np
import os


def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [3,5,63,67]).astype(np.int32)
    res = torch.relu(torch.Tensor(input_x))
    golden = res.numpy().astype(np.int32)
    #print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
