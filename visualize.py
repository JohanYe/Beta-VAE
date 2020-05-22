import numpy as np
import matplotlib.pyplot as plt
import utils
from model import *
import seaborn as sns
import torch.optim as optim
import torchvision
from tqdm import tqdm
from disentanglement_lib import *


sns.set_style("darkgrid")

k = 0
beta = 0.05
batch_size = 64
n_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BetaVAE_conv(latent=6, filters=[128, 64, 64,64], MNIST=False).to(device)
optimizer = optim.Adam(net.parameters(), lr=2e-4)
train_log = {}
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'

# Loading data
train_loader, val_loader = utils.dataloaders(batch_size, MNIST=False)
quick_plot = next(iter(train_loader))
quick_plot = quick_plot[0] if type(quick_plot) is list else quick_plot
quick_plot = quick_plot


utils.load_checkpoint('./checkpoints/best_cluster.pth.tar', net, cpu=True)
quick_plot = next(iter(train_loader))
quick_plot = quick_plot[0] if type(quick_plot) is list else quick_plot
num_traversal = 10
with torch.no_grad():
    images = net.LT_fitted_gauss_2std(quick_plot.to(device), num_var=2, num_traversal=num_traversal)


