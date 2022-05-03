# modified from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..stylegan_op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel // groups, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel // groups * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualConv1d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True,
            bias_init=0, c=1, w=1, init="normal", lr_mul=1
    ):
        super().__init__()
        if init == "normal":
            weight = torch.randn(out_channel, in_channel // groups, kernel_size).div_(lr_mul)
        elif init == "uniform":
            weight = torch.FloatTensor(out_channel, in_channel // groups, kernel_size).uniform_(-1, 1).div_(lr_mul)
        else:
            raise ValueError()
        self.weight = nn.Parameter(weight)
        self.scale = w * c ** 0.5 / math.sqrt(in_channel / groups * kernel_size) * lr_mul

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

        else:
            self.bias = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    @property
    def memory_cost(self):
        return self.out_channel

    @property
    def flops(self):
        f = 2 * self.in_channel * self.out_channel // self.groups - self.out_channel
        if self.bias is not None:
            f += self.out_channel
        return f

    def forward(self, input):
        out = F.conv1d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class NormalizedConv1d(nn.Module):
    # stylegan2 like normalization
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True,
            c=1, w=1, init="normal", lr_mul=1
    ):
        super().__init__()

        self.init = init
        self.scale = w * c ** 0.5 * lr_mul
        if init == "normal":
            self.weight = nn.Parameter(
                torch.randn(out_channel, in_channel // groups, kernel_size)
            )
        elif init == "uniform":
            weight = torch.FloatTensor(out_channel, in_channel // groups, kernel_size).uniform_(-1, 1).div_(lr_mul)
            self.weight = nn.Parameter(weight)
        else:
            raise ValueError()

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    @property
    def memory_cost(self):
        return self.out_channel

    @property
    def flops(self):
        f = 2 * self.in_channel * self.out_channel // self.groups - self.out_channel
        if self.bias is not None:
            f += self.out_channel
        return f

    def forward(self, input):
        scale = self.scale * torch.rsqrt(self.weight.pow(2).sum([1, 2], keepdim=True) + 1e-8)
        if self.init == "uniform":
            scale = scale / 3 ** 0.5  # std of uniform = std of normal / 3**0.5
        out = F.conv1d(
            input,
            self.weight * scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, w=1
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (w / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

        self.in_dim = in_dim
        self.out_dim = out_dim

    @property
    def memory_cost(self):
        return self.out_dim

    @property
    def flops(self):
        f = 2 * self.in_dim * self.out_dim - self.out_dim
        if self.bias is not None:
            f += self.out_dim
        return f

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            assert self.bias is not None
            bias = self.bias * self.lr_mul
            out = fused_leaky_relu(out, bias)

        else:
            bias = None if self.bias is None else self.bias * self.lr_mul
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out



class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out



class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out