import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from matplotlib import gridspec
import imageio


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


class UnFlatten(nn.Module):
    def __init__(self, c_out):
        super(UnFlatten, self).__init__()
        self.c_out = c_out

    def forward(self, input):
        # [Batch, Channels, Width, Height]
        input = input.view(input.size(0), self.c_out, -1)
        # check that it's a perfect square (kinda, floating point precision might make this wrong)
        assert input.size(2) == isqrt(input.size(2)) ** 2
        dim = int(np.sqrt(input.size(2)))
        return input.view(input.size(0), self.c_out, dim, dim)


def save_checkpoint(state, save_dir, ckpt_name='best.pth.tar'):
    file_path = os.path.join(save_dir, ckpt_name)
    if not os.path.exists(save_dir):
        print("Save directory dosen't exist! Makind directory {}".format(save_dir))
        os.mkdir(save_dir)

    torch.save(state, file_path)


def load_checkpoint(checkpoint, model, cpu=False):
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
    if cpu:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)
    saved_dict = checkpoint['state_dict']
    new_dict = model.state_dict()
    new_dict.update(saved_dict)
    model.load_state_dict(new_dict)


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def dataloaders(batch_size, MNIST=True):
    if MNIST:
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True)

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True)
        return train_loader, val_loader

    else:
        root = './data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        # train_set, val_set = data[:int(data.size(0)*0.8)], data[int(data.size(0)*0.8):]
        rand_perm = torch.randperm(data.size(0))
        train_set, val_set = data[rand_perm[:50000]], data[rand_perm[50000:60000]]
        train_kwargs = {'data_tensor': train_set}
        val_kwargs = {'data_tensor': val_set}
        dset = CustomTensorDataset
        train_data = dset(**train_kwargs)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        val_data = dset(**val_kwargs)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        return train_loader, val_loader


def traversal_plotting(x, out_loc, num_traversals=10, original_index=0, silent=False):
    """ expects original to be first index """
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
    ax0 = plt.subplot(gs[0])

    original = x[original_index].reshape(64, 64).cpu()
    img = torch.stack(x[1:], dim=0).cpu().view(6 * num_traversals, 1, 64, 64)
    img_grid = torchvision.utils.make_grid(img, nrow=num_traversals)
    ax0.imshow(original)
    ax0.axis('off')
    ax0.set_title('Original')
    ax1 = plt.subplot(gs[1])
    ax1.set_title('traversals')
    ax1.imshow(np.transpose(img_grid, (1, 2, 0)))
    ax1.axis('off')
    plt.savefig(out_loc, bbox_layout='tight')
    if silent:
        plt.close()


def save_animation(images, filename, num_traversal, fps):
    gif = np.array(images.cpu().detach())
    gif = (gif*255).reshape(num_traversal, 64, 64).astype(np.uint8)
    imageio.mimwrite(filename, gif, fps=fps)
