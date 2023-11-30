import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from noise_model.GMM import GMM


# class MaskedConvolution(nn.Module):
#     """Implements a convolution with mask applied on its weights.

#     Parameters
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     mask : torch.tensor
#         Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
#         the convolution should be masked, and 1s otherwise.
#     **kwargs
#         Additional arguments for the convolution.
#     """
#     def __init__(self, in_channels, out_channels, mask, **kwargs):
#         super().__init__()
#         # For simplicity: calculate padding automatically
#         kernel_size = (mask.shape[0], mask.shape[1])
#         dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
#         padding = tuple(dilation * (kernel_size[i] - 1) // 2 for i in range(2))
#         # Actual convolution
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

#         # Mask as buffer => it is no parameter but still a tensor of the module
#         # (must be moved with the devices)
#         self.register_buffer("mask", mask[None, None])

#     def forward(self, x):
#         self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
#         return self.conv(x)


class VerticalConvolution(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, **kwargs):
    #     # Mask out all pixels below. For efficiency, we could also reduce the kernel
    #     # size in height, but for simplicity, we stick with masking here.
    #     mask = torch.ones(kernel_size, kernel_size)
    #     mask[kernel_size // 2 + 1 :, :] = 0

    #     # For the very first convolution, we will also mask the center row
    #     if mask_center:
    #         mask[kernel_size // 2, :] = 0

    #     super().__init__(in_channels, out_channels, mask, **kwargs)
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        mask_center=False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        rf = (dilation * (kernel_size[0] - 1), dilation * (kernel_size[1] - 1))
        padding = (rf[1] // 2, rf[1] // 2, rf[0], 0)
        self.pad = nn.ConstantPad2d(padding, 0)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        kernel_mask = torch.ones((1, 1, kernel_size[0], kernel_size[1]))
        if mask_center:
            kernel_mask[..., -1, :] = 0
        self.register_buffer("kernel_mask", kernel_mask)

    def forward(self, x):
        x = self.pad(x)
        self.conv.weight.data *= self.kernel_mask
        x = self.conv(x)
        return x


class HorizontalConvolution(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, **kwargs):
    #     # Mask out all pixels on the left. Note that our kernel has a size of 1
    #     # in height because we only look at the pixel in the same row.
    #     mask = torch.ones(1, kernel_size)
    #     mask[0, kernel_size // 2 + 1 :] = 0

    #     # For the very first convolution, we will also mask the center pixel
    #     if mask_center:
    #         mask[0, kernel_size // 2] = 0

    #     super().__init__(in_channels, out_channels, mask, **kwargs)
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        mask_center=False,
    ):
        super().__init__()

        rf = dilation * (kernel_size - 1)
        padding = (rf, 0, 0, 0)
        self.pad = nn.ConstantPad2d(padding, 0)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            dilation=dilation,
        )

        kernel_mask = torch.ones((1, 1, 1, kernel_size))
        if mask_center:
            kernel_mask[..., -1] = 0
        self.register_buffer("kernel_mask", kernel_mask)

    def forward(self, x):
        x = self.pad(x)
        self.conv.weight.data *= self.kernel_mask
        x = self.conv(x)
        return x


class GatedConv(nn.Module):
    """A gated activation unit.


    Parameters
    ----------
    n_filters : int
        Number of hidden channels.
    **kwargs
        Additional arguments for the convolutions.

    """

    def __init__(self, n_filters, **kwargs):
        super().__init__()

        self.up_conv = VerticalConvolution(n_filters, 2 * n_filters, **kwargs)
        self.up_conv_1x1 = nn.Conv2d(2 * n_filters, 2 * n_filters, 1)

        self.left_conv = HorizontalConvolution(n_filters, 2 * n_filters, **kwargs)
        self.left_conv_1x1 = nn.Conv2d(n_filters, n_filters, 1)

    def forward(self, up, left):
        up_feat = self.up_conv(up)
        up_tan, up_sig = up_feat.chunk(2, dim=1)
        up = torch.tanh(up_tan) * torch.sigmoid(up_sig)

        left_feat = self.left_conv(left)
        left_feat = left_feat + self.up_conv_1x1(up_feat)
        left_tan, left_sig = left_feat.chunk(2, dim=1)
        left_out = torch.tanh(left_tan) * torch.sigmoid(left_sig)
        left_out = self.left_conv_1x1(left_out)
        left = left_out + left

        return up, left


class PixelCNN(GMM):
    """A CNN with attention gates and autoregressive convolutions

    Parameters
    ----------
    in_channels : int, optional
        The number of input channels. The default is 1.
    n_filters : int, optional
        The number of hidden channels. The default is 128.
    kernel_size : int, optional
        Side length of the convolutional kernel. The default is 5.
    n_gaussians : int, optional
        Number of components in the Gaussian mixture model. The default is 10.
    depth : int, optional
        The number of hidden layers. The default is 5.
    dropout : Float, optional
        The probability of an element being dropped by dropout. The default is 0.2.
    noise_mean : Float, optional
        Mean of the noise samples, used for normalisation of the data. The default is 0.
    noise_std : Float, optional
        Standard deviation of the noise samples, used for normalisation of the data. The default is 1.

    """

    def __init__(
        self,
        in_channels=1,
        n_filters=128,
        kernel_size=5,
        n_gaussians=10,
        depth=5,
        dropout=0.2,
        noise_mean=0,
        noise_std=1,
        lr=1e-4,
    ):
        self.save_hyperparameters()
        super().__init__(n_gaussians, noise_mean, noise_std, lr)

        out_channels = in_channels * n_gaussians * 3

        self.up_inconv = VerticalConvolution(
            in_channels, n_filters, kernel_size, mask_center=True
        )
        self.left_inconv = HorizontalConvolution(
            in_channels, n_filters, kernel_size, mask_center=True
        )
        self.dropout = nn.Dropout(dropout)

        gatedconvs = []
        for i in range(depth):
            gatedconvs.append(GatedConv(n_filters, kernel_size=kernel_size, dilation=2**i))

        self.gatedconvs = nn.ModuleList(gatedconvs)

        self.outconv = nn.Conv2d(n_filters, out_channels, 1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        up = self.up_inconv(x)
        left = self.left_inconv(x)

        for layer in self.gatedconvs:
            up = self.dropout(up)
            left = self.dropout(left)
            up, left = layer(up, left)

        out = self.outconv(F.elu(left))

        return out
