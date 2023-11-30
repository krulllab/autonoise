import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import numpy as np
import pytorch_lightning as pl
import sys
from tqdm import tqdm

class GMM(pl.LightningModule):
    """Gaussian mixture model.

    Contains functions for calculating Gaussian mixture model
    loglikelihood and for autoregressive image sampling.

    Attributes:
        n_gaussians: An integer for the number of components in the Gaussian
        mixture model.
        noise_mean: Float for the mean of the noise samples, used to normalise
        data.
        noise_std: Float for the standard deviation of the noise samples, also
        used to normalise the data

    """
    
    def __init__(self, n_gaussians, noise_mean, noise_std, lr):
        super().__init__()  
        self.n_gaussians = n_gaussians
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.lr = lr
        
    def get_gaussian_params(self, pred):
        blockSize = self.n_gaussians
        means = pred[:,:blockSize,...]
        stds = torch.sqrt(torch.exp(pred[:,blockSize:2*blockSize,...]))
        weights = torch.exp(pred[:,2*blockSize:3*blockSize,...])
        weights = weights / torch.sum(weights,dim = 1, keepdim = True)
        return means, stds, weights   
    
    def loglikelihood(self, x, s=None):
        """Calculates loglikelihood of noise input.

        Passes noise image through network to obtain Gaussian mixture model
        parameters. Uses those parameters to calculate loglikelihood of noise
        image.


        Parameters
        ----------
        x : torch.FloatTensor
            This is a noise sample when training the noise model, but, when using
            the noise model to evaluate the denoiser, this is a noisy image.
        s : torch.FloatTensor
            When training the noise model, this should be left as none. When training
            the denoiser this is the estimated signal and will be subtracted from the
            noisy image to obtain a noise image.

        Returns
        -------
        loglikelihoods : torch.FloatTensor
            The elementwise loglikelihood of the input.

        """
        
        if s is None:
            s = torch.zeros_like(x)

        n = x - s
        
        n = n - self.noise_mean
        n = n / self.noise_std
        
        if self.training:
            pred = self.forward(n)
        else:
            pred = self.forward(n).detach()
            
        means, stds, weights = self.get_gaussian_params(pred)
        likelihoods= -0.5*((means-n)/stds)**2 - torch.log(stds) -np.log(2.0*np.pi)*0.5
        temp = torch.max(likelihoods, dim = 1, keepdim = True)[0].detach()
        likelihoods=torch.exp( likelihoods -temp) * weights
        loglikelihoods = torch.log(torch.sum(likelihoods, dim = 1, keepdim = True))
        loglikelihoods = loglikelihoods + temp 
        return loglikelihoods
    
    def sampleFromMix(self, means, stds, weights):
        num_components = means.shape[1]
        shape = means[:,0,...].shape
        selector = torch.rand(shape, device = means.device)
        gauss = torch.normal(means[:,0,...]*0, means[:,0,...]*0 + 1)
        out = means[:,0,...]*0

        for i in range(num_components):
            mask = torch.zeros(shape)
            mask = (selector<weights[:,i,...]) & (selector>0)
            out += mask* (means[:,i,...] + gauss*stds[:,i,...])
            selector -= weights[:,i,...]
        
        del gauss
        del selector
        del shape
        return out    
    
    @torch.no_grad()
    def sample(self, img_shape):
        """Samples images from the trained autoregressive model.

        Parameters
        ----------
        img_shape : List or tuple
            The shape of the image with format [N, C, H, W], where N is the number
            of images, C is the colour channel, H is the height of the images, W
            is the width of the images.

        Returns
        -------
        torch.FloatTensor
            The generated images.

        """
        # Create empty image
        img = torch.zeros(img_shape, dtype=torch.float).to(self.device)
        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:, :, : h + 1, :])
                    means, stds, weights = self.get_gaussian_params(pred)
                    means = means[:,:,h,w][...,np.newaxis,np.newaxis]
                    stds = stds[:,:,h,w][...,np.newaxis,np.newaxis]
                    weights = weights[:,:,h,w][...,np.newaxis,np.newaxis]
                    samp = self.sampleFromMix(means, stds, weights).detach()
                    img[:, c, h, w] = samp[:,0,0]
                    
        return img*self.noise_std + self.noise_mean
    
    def training_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("train/nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("val/nll", loss, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
