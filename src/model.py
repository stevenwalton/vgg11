import torch
from torch import nn

class VGG(nn.Module):
    '''
    Our implementation of VGG11
    '''
    def __init__(self,
                 inplace=True):
        super(VGG,self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3,64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=inplace),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(64,128, kernel_size=3,
                                                padding=1),
                                      nn.ReLU(inplace=inplace),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(128,256, kernel_size=3,
                                                padding=1),
                                      nn.ReLU(inplace=inplace),
                                      nn.Conv2d(256,256, kernel_size=3,
                                                padding=1),
                                      nn.ReLU(inplace=inplace),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(256,512, kernel_size=3,
                                                padding=1),
                                      nn.ReLU(inplace=inplace),
                                      nn.Conv2d(512,512, kernel_size=3,
                                                padding=1),
                                      nn.ReLU(inplace=inplace),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Sequential(nn.Linear(25088, 4096),
                                    nn.ReLU(inplace=inplace),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=inplace),
                                    nn.Linear(4096, 1000)
                                    )
        self.sig = nn.Sigmoid()

    def forward(self,x):
        '''
        Our forward implementation
        '''
        x = self.features(x)
        x = self.max_pool(x)
        x = x.view(-1)
        x = self.linear(x)
        return self.sig(x)
