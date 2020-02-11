import torch
from torch import optim

import model 

def trainNet(datapath='.',
             nepochs=1,
             learning_rate=0.1,
             batch_size=32,
             cuda=False,
             savedir='./',
             lossPlotName='loss.png',
             ):
    '''
    Our basic training file
    '''

    model = model.VGG()
    print(model())
    #criterion = nn.MSELoss()
    #for epoch in range(nepochs):
    #    net.train()
    #    epoch_loss = 0
    #    for i, (img, label) in enumerate(data):
