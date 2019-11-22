import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['DES_BACKEND']='torch'

import desolver as de
import desolver.backend as D
import tqdm.auto as tqdm
import numpy as np


        
if __name__ == "__main__":
    np.testing.run_module_suite()
