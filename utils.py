import os
import torch
import torch.nn as nn
import numpy as np
import torchvision

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

def cycle_interval(starting_value, num_frames, min_val, max_val):
  """Cycles through the state space in a single cycle."""
  starting_in_01 = ((starting_value - min_val)/(max_val - min_val)).cpu()
  grid = torch.linspace(starting_in_01.item(), starting_in_01.item() + 2., steps=num_frames+1)[:-1]
  grid -= np.maximum(0, 2*grid - 2)
  grid += np.maximum(0, -2*grid)
  return grid * (max_val - min_val) + min_val

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


def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
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

