import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from torchvision import transforms

def plot_loss(array, savename):
    plt.plot(array)
    plt.title(f"Loss Array ({len(array)} epochs)")
    plt.xlabel(f"Epoch")
    plt.ylabel(f"Loss")
    plt.savefig(savename)

def plot_examples(img_arr, pred_arr, class_list, label_arr, savename):
    fig = plt.figure(figsize=(10,10))
    plt.title("Example Images")
    for i, (img, pred, label) in enumerate(zip(img_arr, pred_arr, label_arr)):
        plt.subplot(3,3,i+1)
        plt.axis('off')
        plt.imshow(to_PIL(img))
        p = np.argmax(pred)
        plt.title(f"Actual {class_list[int(label)]}, Predicted {class_list[p]}")

    plt.savefig(savename)

def to_PIL(img):
    to_img = transforms.ToPILImage()
    return to_img(img)

def confusionMatrix(true, pred, classes, savename='confusionMatrix.png'):
    c = confusion_matrix(true, pred, labels=range(len(classes)))
    print(c)
    fig, ax = plt.subplots()
    im = sns.heatmap(c,
                     cmap='viridis',
                     annot=True,
                     fmt='d',
                     cbar=False,
                     square=True,
                     xticklabels=classes,
                     yticklabels=classes,
                     )

    ax.set_title(f"VGG Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(savename)
    plt.clf()

