import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,
                 datapath,
                 imgtype=".png",
                 size=224,
                 ):
        super(myDataset, self).__init__()
        self.size = size

        # Check if our datapath exists and make sure it is formatted correctly
        assert(os.path.exists(datapath)),f"Path {datapath} DOES NOT exist"
        if "/" not in datapath[-1]:
            datapath += "/"
        self.datapath = datapath

        if "." not in imgtype[0]:
            imgtype = "." + imgtype


