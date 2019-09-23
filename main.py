import torch
import sys
sys.path.append('src')
import model

if __name__ == '__main__':
    model = model.VGG()
    a = torch.randn(1,3,224,224)
    o = model(a)
    print(o.size())
