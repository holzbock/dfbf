import torch
import numpy as np
from PIL import Image
import os
import pdb
import pickle5 as pickle


def save_imgs(imgs, mean=None, std=None, logdir=None, name='img'):
    # make output dir
    if logdir is not None:
        root = logdir + 'generated_images/'
    else:
        root = './generated_images/'
    if not os.path.isdir(root):
        os.mkdir(root)

    # Save imgs
    imgs = imgs.detach().cpu().numpy()
    imgs = imgs * std[None, :, None, None] + mean[None, :, None, None]
    imgs = imgs * 255.
    imgs = imgs.astype(np.uint8)
    for i, img in enumerate(imgs):
        img = img.squeeze()
        img = np.swapaxes(img, 0, 2)
        name_img = root + '%s_%i.jpg'%(name, i)
        pil_img = Image.fromarray(img)
        pil_img.save(name_img)


def get_log_dir(root):
    # get all existing dirs
    dirs = os.walk(root)
    dirs = [dir for dir in dirs]
    dirs = dirs[0][1]
    # get number for next log dir
    numbers = [-1,]
    for dir in dirs:
        if 'exp' in dir:
            num = int(dir.replace('exp', ''))
            numbers.append(num)
    last = max(numbers)
    # Make new log dir
    new = last + 1
    log_dir = root + '/exp' + str(new) + '/'
    os.mkdir(log_dir)

    return log_dir


def visualize_pseudo_images(data_path):
    mean = np.array([0.4914, 0.4822, 0.4465]) # CIFAR10
    std = np.array([0.2471, 0.2435, 0.2616]) # CIFAR10
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    images = data['images']
    imgs = images[0]
    imgs = imgs * std[None, :, None, None] + mean[None, :, None, None]
    imgs = imgs * 255.
    Image.fromarray(imgs[0].numpy().transpose(1,2,0).astype(np.uint8)).save('./generated_images/test.jpg')


class PseudoImagesDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)