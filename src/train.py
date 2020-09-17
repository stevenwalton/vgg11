import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torchvision import datasets

import model 

def trainNet(datapath='.',
             nepochs=1,
             learning_rate=0.001,
             batch_size=64,
             cuda=False,
             savedir='./',
             lossPlotName='loss.png',
             num_workers=24,
             ):
    '''
    Our basic training file
    '''

    if cuda:
        print(f"Running on GPU")
        device = torch.device('cuda')
        net = model.VGG().to(device)
    else:
        print(f"Running on CPU")
        device = torch.device('cpu')
        net = model.VGG()
    # Dataset
    #imagenet = datasets.ImageNet('/research/imgnet/ILSVRC2013_DET_train/',
    #        split='train')
    t = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            ])
    train_images = datasets.CIFAR10('.', train=True, download=True, transform=t)
    train_data = torch.utils.data.DataLoader(train_images,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    epoch_loss_array = torch.zeros(nepochs)
    print(f"Train data {len(train_images)}")
    for epoch in range(nepochs):
        net.train()
        epoch_loss = 0
        for i, (img, label) in enumerate(train_data):
            optimizer.zero_grad()
            out = net(img.to(device))
            loss = criterion(out.to(device), label.to(device))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss /= (i+1)
        print(f"Epoch {epoch} loss {epoch_loss}")
        epoch_loss_array[epoch] = epoch_loss

    # Testing
    with torch.no_grad():
        test_loss = 0.
        test_images = datasets.CIFAR10('.', train=False, download=True,
                transform=t)
        test_data = torch.utils.data.DataLoader(test_images,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
        for i,(img, label) in enumerate(test_data):
            optimizer.zero_grad()
            out = net(img.to(device))
            loss = criterion(out.to(device), label.to(device))
            test_loss += loss.item()
        test_loss /= (i+1)
    print(f"Test loss {test_loss}")
