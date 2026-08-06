import numpy as np
import torch
import os
    
os.system("mkdir -p input")
os.system("mkdir -p output")

shape = [3,53,57]
minval = -100
maxval = -1
dtype = torch.float32

def gen_golden_data_simple():

    input_x= torch.rand(shape, dtype=dtype) * (maxval - minval) + minval
    input_x.numpy().tofile("./input/input_x.bin")

    golden = torch.asinh(x=input_x)
    golden.numpy().tofile("./output/golden.bin")

    print("input_x is: ", input_x)
    print("golden is: ", golden)

if __name__ == "__main__":
    gen_golden_data_simple()

