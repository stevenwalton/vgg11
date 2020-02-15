import torch
from torch import nn

def convDown(_in, _out, kernel_size=3,
             padding=1, stride=1, inplace=True,
             maxPool=False, kernel_size_mp=2, stride_mp=2):
    ''' Function Definition for our convolutional layers '''
    layers = [nn.Conv2d(_in,_out, kernel_size=kernel_size, 
                            padding=padding, stride=stride)]
    layers.append(nn.ReLU(inplace=inplace))
    if maxPool:
        layers.append(nn.MaxPool2d(kernel_size=kernel_size_mp,
                                   stride=stride_mp))
    return layers

def linearBlock(_in, _out, last=False,
                inplace=True, p=0.5, inplace_dr=False):
    ''' Blocks for the classifier '''
    layers = [nn.Linear(_in, _out)]
    if not last:
        layers.append(nn.ReLU(inplace=inplace))
        layers.append(nn.Dropout(p=p, inplace=inplace_dr))
    return layers

class VGG(nn.Module):
    '''
    Our implementation of VGG11
    '''
    def __init__(self,
                 inplace=True):
        super(VGG,self).__init__()

        # Making Features
        convSizes = [3,64,128,256,256,512,512,512,512]
        poolingBlock = [True,True,False,True,False,True,False,True]

        features = []
        for i in range(len(convSizes)-1):
            features += convDown(convSizes[i], convSizes[i+1],
                maxPool=poolingBlock[i])
        self.features = nn.Sequential(*features)

        # Making avgpool
        adaptivePoolSize   = (7,7) 
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=adaptivePoolSize)

        # Making classifier
        classificationSize = 1000
        linearSizes = [(adaptivePoolSize[0]*adaptivePoolSize[1])*convSizes[-1], 4096, 4096,
                       classificationSize]
        classifier = []
        for i in range(len(linearSizes)-1):
            classifier += linearBlock(linearSizes[i], linearSizes[i+1],
                (i+1 == len(linearSizes)-1))
        self.classifier = nn.Sequential(*classifier)

        '''
        Alternatively, we could write this more explicitly but the above is more
        modular and easier to edit

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

        self.avgpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                    nn.ReLU(inplace=inplace),
                                    nn.Dropout(p=0.5, inplace=inplace),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=inplace),
                                    nn.Dropout(p=0.5, inplace=inplace),
                                    nn.Linear(4096, 1000)
                                    )
        '''

    def forward(self,x):
        '''
        Our forward implementation
        '''
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1)
        x = self.classifier(x)
        return x
