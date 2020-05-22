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
net = BetaVAE_conv(latent=6, filters=[128, 64, 64], MNIST=False).to(device)
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
img_grid = torchvision.utils.make_grid(quick_plot.cpu(), nrow=8).numpy()
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.axis('off')
plt.savefig('./figures/Figure_1.pdf')

# Training loop
for epoch in range(n_epochs):
    train_batch_loss = []
    for batch in tqdm(train_loader):
        net.train()
        batch = batch[0] if type(batch) is list else batch
        batch = batch.to(device)
        loss, kl, nll = net.calc_loss(batch, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log[k] = [loss.item(), kl.item(), nll.item()]
        train_batch_loss.append(loss.item())

        k += 1
        if beta < 3 and k > 30000:
            beta += 0.0005

    val_batch_loss = []
    for batch in tqdm(val_loader):
        batch = batch[0] if type(batch) is list else batch
        with torch.no_grad():
            net.eval()
            loss, kl, nll = net.calc_loss(batch.to(device), 1)
            val_log[k] = [loss.item(), kl.item(), nll.item()]
            val_batch_loss.append(loss.item())

    if loss.item() < best_nll:
        best_nll = loss.item()
        utils.save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch + 1, n_epochs, k, np.mean(train_batch_loss), np.mean(val_batch_loss)))

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
ax[0].set_title('ELBO Training Curve')
ax[0].legend(loc='best')
ax[0].set_xlabel('Num Steps')

ax[1].plot(train_x_vals, kl_train, label='Training KL')
ax[1].plot(x_val, kl_val, label='Validation KL')
ax[1].set_title('Kullback-Leibler Divergence Curve')
ax[1].legend(loc='best')
ax[1].set_xlabel('Num Steps')

ax[2].plot(train_x_vals, recon_train, label='Training Reconstruction Error')
ax[2].plot(x_val, recon_val, label='Validation Reconstruction Error')
ax[2].set_title('Reconstruction Error Curve')
ax[2].legend(loc='best')
ax[2].set_xlabel('Num Steps')
plt.savefig('./figures/Figure_2.pdf', bbox_inches='tight')
# plt.close()


quick_plot = next(iter(train_loader))
quick_plot = quick_plot[0] if type(quick_plot) is list else quick_plot
quick_plot = quick_plot

utils.load_checkpoint('./checkpoints/best.pth.tar', net)
test = net(quick_plot.to(device)).detach().cpu()
img_grid = torchvision.utils.make_grid(test.reshape(test.shape[0], 1, 64, 64), nrow=8)
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.axis('off')




