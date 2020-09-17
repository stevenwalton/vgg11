import torch
import sys
sys.path.append('src')

import train

if __name__ == '__main__':
    train.trainNet(cuda=True,
                   nepochs=100)
