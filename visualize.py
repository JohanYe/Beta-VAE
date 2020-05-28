import numpy as np
import matplotlib.pyplot as plt
import utils
from model import *
import seaborn as sns
import torch.optim as optim
import torchvision
from disentanglement_lib import *

sns.set_style("darkgrid")

k = 0
beta = 0.05
batch_size = 64
n_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = BetaVAE_conv(latent=6, filters=[128, 64, 64, 64], MNIST=False).to(device)
net = BetaVAE_conv(latent=10, filters=[32, 32, 64, 64], MNIST=False).to(device)
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

utils.load_checkpoint('./checkpoints/best.pth.tar', net)
img_grid = torchvision.utils.make_grid(quick_plot.cpu(), nrow=8).numpy()
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.axis('off')
plt.savefig('./figures/Figure_1.pdf')

test = net(quick_plot.to(device)).detach().cpu()
img_grid = torchvision.utils.make_grid(test.reshape(test.shape[0], 1, 64, 64), nrow=8)
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.axis('off')
plt.savefig('./figures/Figure_3.pdf', bbox_inches='tight')

# net = nn.DataParallel(net)
# utils.load_checkpoint('./checkpoints/best_cluster.pth.tar', net)
utils.load_checkpoint('./checkpoints/best.pth.tar', net)
quick_plot = next(iter(train_loader))
quick_plot = quick_plot[0] if type(quick_plot) is list else quick_plot
num_traversal = 10

with torch.no_grad():
    net.eval()
    # images = net.module.LT_fitted_gauss_2std(quick_plot.to(device), num_var=10, num_traversal=num_traversal)
    images = net.LT_fitted_gauss_2std(quick_plot.to(device), num_var=10, num_traversal=num_traversal,silent=True)

# Getting my and latents_values
mu_train, latent_train, mu_val, latent_val = utils.get_mu_and_latents(net, batch_size=batch_size, seed=0)

# Rescaling to classes
latent_train, latent_val = rescale_dsprites_latents(latent_train, latent_val)

# calculate importance matrix of GBT, may take a long time
scores, importance_matrix = compute_dci(mu_train, latent_train, mu_val, latent_val, load=False)

cmap = sns.cubehelix_palette(as_cmap=True)
num_plot = 1000
c = mu_train @ importance_matrix
for i in range(latent_train.shape[1]):
    fig, ax = plt.subplots()

    # Plot prep
    plot_x = np.expand_dims(np.arange(1, 11),axis=0).repeat(num_plot,axis=0)
    offset = np.random.uniform(-0.3,0.3, size=(num_plot, 10))
    plot_x = plot_x + offset
    c_plot = np.expand_dims(c[:num_plot, i], axis=1).repeat(mu_train.shape[1], axis=1)
    g = ax.scatter(plot_x, mu_train[:num_plot, :], c=c_plot, s=10)
    ax.set_xlabel('Mu')
    ax.set_ylabel('Mean of q(z|x)')
    title = utils.plot_tiles_dsprites(i)
    plt.title(title)
    plt.colorbar(g)
    filename = './figures/DCI/latent_importance_' + str(i) + '.pdf'
    plt.savefig(filename, bbox_inches='tight')

