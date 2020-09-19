import numpy as np
import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torchvision import datasets

import model 
import utils

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
    if "/" not in savedir[-1]:
        savedir += "/"
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

    utils.plot_loss(epoch_loss_array, savedir + lossPlotName)
    net.load_state_dict(torch.load('final_model.pt'))

    # Testing
    with torch.no_grad():
        net.eval()
        test_loss = 0.
        test_images = datasets.CIFAR10('.', train=False, download=True,
                transform=t)
        test_data = torch.utils.data.DataLoader(test_images,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck']
        corrects = np.zeros(len(classes),dtype=np.int)
        totals = np.zeros(len(classes),dtype=np.int)
        trues = np.zeros(len(test_images))
        preds = np.zeros(len(test_images))
        for i,(img, label) in enumerate(test_data):
            optimizer.zero_grad()
            out = net(img.to(device))
            trues[i*batch_size:(i+1)*batch_size] = label.to('cpu').numpy()
            preds[i*batch_size:(i+1)*batch_size] = [np.argmax(o) for o in out.to('cpu').numpy()]
            for o,l in zip(out, label):
                o = o.to('cpu').numpy()
                l = l.to('cpu').numpy()
                totals[l] += 1
                if np.argmax(o) == l:
                    corrects[l] += 1
            loss = criterion(out.to(device), label.to(device))
            test_loss += loss.item()
            if i == 0:
                utils.plot_examples(img[:9].to('cpu'),
                                    out[:9].to('cpu').numpy(), 
                                    classes, 
                                    label[:9].to('cpu').numpy(),
                                    savedir + "examples.png")
        test_loss /= (i+1)
    print(f"Test loss {test_loss}")
    utils.confusionMatrix(trues, preds, classes)
    torch.save(net.state_dict(), "final_model.pt")
    #print(f"Corrects {corrects}")
    #print(f"Totals {totals}")
    print("Accuracy")
    print(f"Name\tCorrects\tAccuracy")
    for i in range(len(classes)):
        print(f"{classes[i]}\t{corrects[i]}\t\t{corrects[i]/totals[i]}")
    print(30*"-")
    print(f"Sum\t{np.sum(corrects)}\t\t{np.sum(corrects)/np.sum(totals)}")
