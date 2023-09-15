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



class IWAE(nn.Module):
    ''' Importance-Weighted Autoencoder (IWAE) is an extension of the Variational Autoencoder (VAE) neural network architecture
    used for generative modeling. It enhances VAEs by enabling the generation of multiple diverse samples from the latent space,
    providing a more comprehensive representation of complex data distributions.

    Parameters
    ----------
    input_size : int
        The size of the input data.

    hidden_size : int
        The size of the hidden layer in the encoder and decoder networks.

    latent_size : int
        The size of the latent space.

    K : int
        The number of importance-weighted samples.

    Methods
    -------
    encode(x)
        Encodes input data into the latent space and computes mean and log-variance.

    decode(z)
        Decodes latent space representations into reconstructed data.

    reparameterize(mu, logvar)
        Reparameterizes the latent space to sample from the learned distribution.

    forward(x)
        The forward pass of the IWAE, combining encoding, reparameterization, and decoding.

    loss_function(x_hat, x, mu, logvar, z)
        Computes the IWAE loss, including the reconstruction loss and importance-weighted terms.

    plot()
        Generates and plots random samples from the IWAE.
    '''

    def __init__(self, input_size, hidden_size, latent_size, K):
        super(IWAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.K = K

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
            A point in the latent space of shape (num_sample, batch_size, latent_size).

        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed output data of shape (num_sample, batch_size, input_size).
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
            A sample from the latent space using the reparameterization trick
            of shape (num_sample, batch_size, latent_size).
        '''

        std = torch.exp(0.5 * logvar)
        # Repeat the std along the first axis to sample multiple times
        dims = (self.K,) + (std.shape)
        eps = torch.randn(*dims, device=mu.device)
        z = mu + eps * std
        return z

    def forward(self, x):
        ''' The forward pass of the IWAE, combining encoding, reparameterization, and decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, input_size).

        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed output data of shape (num_sample, batch_size, input_size).

        loss : torch.Tensor
            The IWAE loss.
        '''

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        loss = self.loss_function(x_hat, x, mu, logvar, z)
        return x_hat, loss

    def loss_function(self, x_hat, x, mu, logvar, z):
        ''' Computes the IWAE loss, including the reconstruction loss and importance-weighted terms.

        Parameters
        ----------
        x_hat : torch.Tensor
            The reconstructed output data of shape (num_sample, batch_size, input_size).

        x : torch.Tensor
            Input data of shape (batch_size, input_size).

        mu : torch.Tensor
            The mean of the latent space distribution.

        logvar : torch.Tensor
            The log variance of the latent space distribution.

        z : torch.Tensor
            Sampled points in the latent space of shape (num_sample, batch_size, latent_size).

        Returns
        -------
        loss : torch.Tensor
            The total IWAE loss.
        '''

        std = torch.exp(0.5 * logvar)
        # q(z|x)
        log_q_z_x = torch.distributions.Normal(loc=mu, scale=std).log_prob(z).sum(-1)
        # p(z)
        log_p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)).log_prob(z).sum(-1)
        # p(x|z)
        xx = x.unsqueeze(0).repeat(self.K,1,1)
        likelihood = - F.binary_cross_entropy(x_hat, xx, reduction='none').sum(-1)

        log_weight = log_p_z + likelihood - log_q_z_x
        log_weight2 = log_weight - torch.max(log_weight, 0)[0]  # for stability in original implementation
        weight = torch.exp(log_weight2)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()

        loss = -torch.sum(torch.sum(weight * log_weight, 0)) + torch.log(torch.Tensor([self.K]))
        return loss


    def plot(self):
        ''' Generates and plots random samples from the IWAE. '''

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