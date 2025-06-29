import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import numpy as np
from torchvision.utils import make_grid as make_image_grid

def check_gpu_cuda_works():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    devNumber = torch.cuda.current_device()
    print("Current device number is:", devNumber)
    devName = torch.cuda.get_device_name(devNumber)
    print("Current device name is:", devName)
    return device


def sample_normal(mean, logvar):
    sd = torch.exp(logvar * 0.5)
    # e = Variable(torch.randn(sd.size())) # Sample from standard normal
    e = torch.randn_like(sd)
    z = e.mul(sd).add_(mean)
    return z


class VAE(nn.Module):
    #def __init__(self, latent_dim=32, input_height=227, input_width=227):
    def __init__(self, latent_dim=32, input_height=227, input_width=227):

        super(VAE, self).__init__()

        self.input_height = input_height
        self.input_width = input_width

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 114, 114)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 57, 57)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 29, 29)
            nn.ReLU(),
            #nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> (256, 15, 15)  or rather 256, 14, 14 (?)
            #nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            dummy_output = self.encoder_conv(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)

        # ----- Latent mapping -----
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ----- Decoder -----
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_deconv = nn.Sequential(
            #nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 30, 30)
            #nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (64, 60, 60)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> (32, 120, 120)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # -> (1, 240, 240)
            nn.Sigmoid()
        )

    def encoder(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flatten_dim)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decoder(self, z):
        #z = self.fc_z(z).view(-1, 256, 15, 15)
        z = self.fc_z(z).view(-1, 256, self.input_height // 16, self.input_width // 16)  # based on 4x stride-2 layers
        x_out = self.decoder_deconv(z)
        x_out = x_out[:, :, :self.input_height, :self.input_width]  # center crop to match original input
        return x_out

    def forward(self, x_in):
        z_mu, z_logvar = self.encoder(x_in)
        z = sample_normal(z_mu, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mu, z_logvar


def criterion_with_weighted_kl(x_out, x_in, z_mu, z_logvar, kl_weight):
    """
    BCE + weighted KL divergence VAE loss.

    Args
    ----
    x_out : Tensor               # decoder output (logits or probs)
    x_in  : Tensor               # original input
    z_mu  : Tensor               # latent mean
    z_logvar : Tensor           # latent log-variance
    kl_weight : float           # weight applied to the KL term

    Returns
    -------
    loss : Tensor               # scalar, averaged over batch
    """
    # Reconstruction term (sum over all pixels, then we’ll divide by batch)
    bce_loss = F.binary_cross_entropy(x_out, x_in, reduction='sum')

    # KL term (sum over latent dims *and* batch)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    # Combine with weighting and normalise by batch size
    loss = (bce_loss + kl_weight * kld_loss) / x_out.size(0)
    return loss


def train(model, optimizer, dataloader, epochs):
    losses = []
    for epoch in trange(epochs, desc='Epochs'):
        for images, _ in dataloader:
            x_in = Variable(images).to(device, non_blocking=True)
            # x_in = images.to(device)

            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss = criterion_with_weighted_kl(x_out, x_in, z_mu, z_logvar, kl_weight=1.0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses


def test(model, dataloader):
    running_loss = 0.0
    for images, _ in dataloader:
        x_in = Variable(images).to(device, non_blocking=True)
        x_out, z_mu, z_logvar = model(x_in)
        loss = criterion_with_weighted_kl(x_out, x_in, z_mu, z_logvar, kl_weight=1.0)
        running_loss = running_loss + (loss.item() * x_in.size(0))  # Korrektur hier
    return running_loss / len(dataloader.dataset)

def pil_loader(path):
    return Image.open(path).convert("L")  # "L" for grayscale


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),  # resize from 227 to keep symmetrical for decoder
        transforms.ToTensor()])

    train_dataset = datasets.DatasetFolder(
        root="./Concrete/Data/non-crack",
        loader=pil_loader,
        extensions=('jpg', 'jpeg', 'png', 'bmp'),
        transform=transform
    )

    test_dataset = datasets.DatasetFolder(
        root="./Concrete/Data/crack",
        loader=pil_loader,
        extensions=('jpg', 'jpeg', 'png', 'bmp'),
        transform=transform
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return trainloader, testloader


def visualize_losses_moving_average(losses,window=50,boundary='valid',ylim=(95,125)):
    mav_losses = np.convolve(losses,np.ones(window)/window,boundary)
    corrected_mav_losses = np.append(np.full(window-1,np.nan),mav_losses)
    plt.figure(figsize=(10,5))
    plt.plot(losses)
    plt.plot(corrected_mav_losses)
    plt.ylim(ylim)
    plt.show()


def visualize_concrete_vae(model, dataloader, title, num=16):
    def imshow(img):
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(title)
        plt.show()

    images, _ = next(iter(dataloader))
    images = images[0:num, :, :]
    x_in = Variable(images).to(device, non_blocking=True)
    x_out, _, _ = model(x_in)
    x_out = x_out.data
    imshow(make_image_grid(images))
    imshow(make_image_grid(x_out))

if __name__ == '__main__':
    device = check_gpu_cuda_works()
    model = VAE()
    model.to(device)

    trainloader, testloader = get_dataloaders()
    dataiter = iter(trainloader)
    images, _ = next(dataiter)

    optimizer = torch.optim.Adam(model.parameters())
    train_losses = train(model, optimizer, trainloader, epochs=15)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.show()
    visualize_losses_moving_average(train_losses)

    test_loss = test(model, testloader)
    print(f"Test loss: {test_loss}")
    visualize_concrete_vae(model, testloader, title="Test Reconstruction")

    train_loss = test(model, trainloader)
    print(f"Train loss: {train_loss}")
    visualize_concrete_vae(model, trainloader, title="Train Reconstruction")
