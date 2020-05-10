import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import *
import seaborn as sns
import torch.optim as optim
import torchvision
from tqdm import tqdm

sns.set_style("darkgrid")

k = 0
beta = 0
batch_size = 64
n_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BetaVAE_Linear().to(device)
optimizer = optim.Adam(net.parameters(), lr=2e-4)
train_log = {}
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'

# Loading data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)

quick_plot, _ = next(iter(train_loader))
quick_plot = (quick_plot)
img_grid = torchvision.utils.make_grid(quick_plot.cpu(), nrow=8).numpy()
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.imshow(img_grid)
plt.axis('off')
plt.savefig('./figures/Figure_1.pdf')

# Training loop
for epoch in range(n_epochs):
    train_batch_loss = []
    for batch, label in tqdm(train_loader):
        net.train()
        batch = batch.to(device)
        loss, kl, nll = net.calc_loss(batch, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log[k] = [loss.item(), kl.item(), nll.item()]
        train_batch_loss.append(loss.item())

        k += 1

    val_batch_loss = []
    for batch, label in tqdm(val_loader):
        with torch.no_grad():
            net.eval()
            loss, kl, nll = net.calc_loss(batch.to(device), 1)
            val_log[k] = [loss.item(), kl.item(), nll.item()]
            val_batch_loss.append(loss.item())

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch + 1, n_epochs, k, np.mean(train_batch_loss), np.mean(val_batch_loss)))

test = net(quick_plot.to(device)).detach().cpu()
img_grid = torchvision.utils.make_grid(test.reshape(test.shape[0], 1, 28, 28), nrow=8)
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.axis('off')

# Plotting each minibatch step
x_val = list(val_log.keys())
values = np.array(list(val_log.values()))
loss_val, kl_val, recon_val = values[:, 0], values[:, 1], values[:, 2]

# Plot the loss graph
train_x_vals = np.arange(len(train_log))
values = np.array(list(train_log.values()))
loss_train, kl_train, recon_train = values[:, 0], values[:, 1], values[:, 2]

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].plot(train_x_vals, loss_train, label='Training ELBO')
ax[0].plot(x_val, loss_val, label='Validation ELBO')
ax[0].set_title('Training Curve')
ax[1].plot(train_x_vals, kl_train, label='Training KL')
ax[1].plot(x_val, kl_val, label='Validation KL')
ax[1].set_title('Kullback-Leibler Divergence Curve')
ax[2].plot(train_x_vals, recon_train, label='Training Reconstruction Error')
ax[2].plot(x_val, recon_val, label='Validation Reconstruction Error')
ax[2].set_title('Reconstruction Error Curve')
plt.legend(loc='best')

plt.xlabel('Num Steps')
plt.ylabel('NLL in bits per dim')
# plt.savefig('./Hw3/Figures/Figure_8.pdf', bbox_inches='tight')
# plt.close()
