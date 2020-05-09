import os
import torch
import torch.nn as nn


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

def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    saved_dict = checkpoint['state_dict']
    new_dict = model.state_dict()
    new_dict.update(saved_dict)
    model.load_state_dict(new_dict)