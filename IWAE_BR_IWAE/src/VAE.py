import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler




class VAE(nn.Module):
    ''' Variational Autoencoder (VAE) for generative modeling.

    Parameters
    ----------
    input_size : int
        The size of the input data.

    hidden_size : int
        The size of the hidden layer in the encoder and decoder networks.

    latent_size : int
        The size of the latent space.

    Methods
    -------
    encode(x)
        Encodes input data into the latent space and computes mean and log-variance.

    decode(z)
        Decodes latent space representations into reconstructed data.

    reparameterize(mu, logvar)
        Reparameterizes the latent space to sample from the learned distribution.

    forward(x)
        The forward pass of the VAE, combining encoding, reparameterization, and decoding.

    loss(x_hat, x, mu, logvar)
        Computes the VAE loss, including the reconstruction loss and KL divergence.

    plot()
        Generates and plots random samples from the VAE.
    '''

    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.fc_encode = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.fc_decode1 = nn.Linear(latent_size, hidden_size)
        self.fc_decode2 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        ''' Encodes input data into the latent space and computes mean and log-variance.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, input_size).

        Returns
        -------
        mu : torch.Tensor
            The mean of the latent space distribution for the input data x.

        logvar : torch.Tensor
            The log variance of the latent space distribution for the input data x.
        '''

        h = F.relu(self.fc_encode(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        ''' Decodes latent space representations into reconstructed data.

        Parameters
        ----------
        z : torch.Tensor
            A point in the latent space of shape (batch_size, latent_size).

        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed output data of shape (batch_size, input_size).
        '''

        h = F.relu(self.fc_decode1(z))
        x_hat = torch.sigmoid(self.fc_decode2(h))
        return x_hat

    def reparameterize(self, mu, logvar):
        ''' Reparameterizes the latent space to sample from the learned distribution.

        Parameters
        ----------
        mu : torch.Tensor
            The mean of the latent space distribution.

        logvar : torch.Tensor
            The log variance of the latent space distribution.

        Returns
        -------
        z : torch.Tensor
            A sample from the latent space using the reparameterization trick.
        '''

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        ''' The forward pass of the VAE, combining encoding, reparameterization, and decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, input_size).

        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed output data of shape (batch_size, input_size).

        mu : torch.Tensor
            The mean of the latent space distribution for the input data x.

        logvar : torch.Tensor
            The log variance of the latent space distribution for the input data x.
        '''

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def loss(self, x_hat, x, mu, logvar):
        ''' Computes the VAE loss, including the reconstruction loss and Kullback-Leibler (KL) divergence.

        Parameters
        ----------
        x_hat : torch.Tensor
            The reconstructed output data of shape (batch_size, input_size).

        x : torch.Tensor
            Input data of shape (batch_size, input_size).

        mu : torch.Tensor
            The mean of the latent space distribution.

        logvar : torch.Tensor
            The log variance of the latent space distribution.

        Returns
        -------
        loss : torch.Tensor
            The total loss, which is a combination of the reconstruction loss and the KL divergence loss.
        '''

        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='none').sum(-1).sum()
        #reconstruction_loss = -torch.distributions.Bernoulli(x_hat).log_prob(x).sum(-1).sum()
        kl_divergence = torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=-1))
        return reconstruction_loss + kl_divergence

    def plot(self):
        ''' Generates and plots random samples from the VAE. '''

        # Generating samples
        with torch.no_grad():
            random_samples = torch.randn(16, self.latent_size)
            generated_samples = self.decode(random_samples)

        # Reshape generated samples
        generated_samples = generated_samples.view(-1, 28, 28).numpy()

        # Plot generated samples
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_samples[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

