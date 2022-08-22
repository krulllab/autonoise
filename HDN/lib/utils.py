import torch
import numpy as np
from torch import nn
import torch.functional as F
from torchvision.utils import save_image
import os

class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self,
                 size=None,
                 scale=None,
                 mode='bilinear',
                 align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(x,
                            size=self.size,
                            scale_factor=self.scale,
                            mode=self.mode,
                            align_corners=self.align_corners)
        return out

def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x, size, mode) -> torch.Tensor:
    """ Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, h, w) to new height
    and width given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """

    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]
    

def free_bits_kl(kl,
                 free_bits,
                 batch_average = False,
                 eps = 1e-6) -> torch.Tensor:
    """Computes free-bits version of KL divergence.
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).
    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)
    Returns:
        The KL with free bits
    """

    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)

def pad_img_tensor(x, size) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')

def img_grid_pad_value(imgs, thresh = .2) -> float:
    """Returns padding value (black or white) for a grid of images.
    Hack to visualize boundaries between images with torchvision's
    save_image(). If the median border value of all images is below the
    threshold, use white, otherwise black (which is the default).
    Args:
        imgs (torch.Tensor): A 4d tensor
        thresh (float, optional): Threshold in (0, 1).
    Returns:
        pad_value (float): The padding value
    """

    assert imgs.dim() == 4
    imgs = imgs.clamp(min=0., max=1.)
    assert 0. < thresh < 1.

    imgs = imgs.mean(1)  # reduce to 1 channel
    h = imgs.size(1)
    w = imgs.size(2)
    borders = list()
    borders.append(imgs[:, 0].flatten())
    borders.append(imgs[:, h - 1].flatten())
    borders.append(imgs[:, 1:h - 1, 0].flatten())
    borders.append(imgs[:, 1:h - 1, w - 1].flatten())
    borders = torch.cat(borders)
    if torch.median(borders) < thresh:
        return 1.0
    return 0.0
    
def save_image_grid(images,filename,nrows):
    """Saves images on disk.
    Args:
        images (torch.Tensor): A 4d tensor
        filename (string): Threshold in (0, 1)
        nrows (int): Number of rows in the image grid to be saved.
    """
    pad = img_grid_pad_value(images)
    save_image(images, filename, nrow=nrows, pad_value=pad, normalize=True)
    
def generate_and_save_samples(model, filename, nrows = 4) -> None:
    """Save generated images at intermediate training steps.
       Args:
           model: instance of LadderVAE class
           filename (str): filename where to save denoised images
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """
    samples = model.sample_prior(nrows**2)
    save_image_grid(samples, filename, nrows=nrows)
    return samples
    
def save_image_grid_reconstructions(inputs,recons,filename):
    assert inputs.shape == recons.shape
    n_img = inputs.shape[0]
    n = int(np.sqrt(2 * n_img))
    imgs = torch.stack([inputs.cpu(), recons.cpu()])
    imgs = imgs.permute(1, 0, 2, 3, 4)
    imgs = imgs.reshape(n**2, inputs.size(1), inputs.size(2), inputs.size(3))
    save_image_grid(imgs, filename, nrows=n)
    
def generate_and_save_reconstructions(x,filename,model,nrows) -> None:
    """Save denoised images at intermediate training steps.
       Args:
           x (Torch.tensor): Batch of images from test set
           filename (str): filename where to save denoised images
           model: instance of LadderVAE class
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """
    n_img = nrows**2 // 2
    if x.shape[0] < n_img:
        msg = ("{} data points required, but given batch has size {}. "
               "Please use a larger batch.".format(n_img, x.shape[0]))
        raise RuntimeError(msg)
    
    outputs = model.forward(x)

    # Try to get reconstruction from different sources in order
    recons = None
    possible_recons_names = ['out_recons', 'out_mean', 'out_sample']
    for key in possible_recons_names:
        try:
            recons = outputs[key]
            if recons is not None:
                break  # if we found it and it's not None
        except KeyError:
            pass
    if recons is None:
        msg = ("Couldn't find reconstruction in the output dictionary. "
               "Tried keys: {}".format(possible_recons_names))
        raise RuntimeError(msg)

    # Pick required number of images
    x = x[:n_img]
    recons = recons[:n_img]

    # Save inputs and reconstructions in a grid
    save_image_grid_reconstructions(x, recons, filename)
    
def save_images(x, img_folder, model, nrows, step) -> None:
    """Save generated images and denoised images at intermediate training steps.
       Args:
           img_folder (str): Folder where to save images
           model: instance of LadderVAE class
           test_loader: Test loader used to denoise during intermediate traing steps.
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """

    # Save model samples
    fname = os.path.join(img_folder, 'sample_' + str(step) + '.png')
    generate_and_save_samples(model, fname, nrows)

    # Save model original/reconstructions
    fname = os.path.join(img_folder, 'reconstruction_' + str(step) + '.png')

    generate_and_save_reconstructions(x, fname, model, nrows)
    
def plotProbabilityDistribution(signalBinIndex, histogram, gaussianMixtureNoiseModel, min_signal, max_signal, n_bin, device):
    """Plots probability distribution P(x|s) for a certain ground truth signal. 
       Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
        Parameters
        ----------
        signalBinIndex: int
            index of signal bin. Values go from 0 to number of bins (`n_bin`).
        histogram: numpy array
            A square numpy array of size `nbin` times `n_bin`.
        gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
            Object containing trained parameters.
        min_signal: float
            Lowest pixel intensity present in the actual sample which needs to be denoised.
        max_signal: float
            Highest pixel intensity present in the actual sample which needs to be denoised.
        n_bin: int
            Number of Bins.
        device: GPU device
        """
    histBinSize=(max_signal-min_signal)/n_bin
    querySignal_numpy= (signalBinIndex/float(n_bin)*(max_signal-min_signal)+min_signal)
    querySignal_numpy +=histBinSize/2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)
    
    queryObservations_numpy=np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy+=histBinSize/2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch=gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy=pTorch.cpu().detach().numpy()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Observation Bin')
    plt.ylabel('Signal Bin')
    plt.imshow(histogram**0.25, cmap='gray')
    plt.axhline(y=signalBinIndex+0.5, linewidth=5, color='blue', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(queryObservations_numpy, histogram[signalBinIndex, :]/histBinSize, label='GT Hist: bin ='+str(signalBinIndex), color='blue', linewidth=2)
    plt.plot(queryObservations_numpy, pNumpy, label='GMM : '+' signal = '+str(np.round(querySignal_numpy,2)), color='red',linewidth=2)
    plt.xlabel('Observations (x) for signal s = ' + str(querySignal_numpy))
    plt.ylabel('Probability Density')
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()