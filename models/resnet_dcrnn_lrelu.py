import math
import torch
from functools import reduce
from operator import __add__
from timm.models.layers import trunc_normal_, DropPath
from .convnextv2_utils import LayerNorm, GRN

from .layers import Output


class conv3d_padding_same(torch.nn.Module):
    """
    Padding so that the next Conv3d layer outputs an array with the same dimension as the input.
    Depth, height and width are the kernel dimensions.
    
    Observe: if kernel_size[i] == 1, then (logically) no padding is applied in i^{th} dimension.

    Example:
    batch_size = 8
    in_channels = 3
    out_channel = 16
    kernel_size = (2, 3, 5)
    stride = 1  # could also be 2, or 3, etc.
    pad_value = 0
    conv = torch.nn.Conv3d(in_channels, out_channel, kernel_size, stride=stride)

    x = torch.empty(batch_size, in_channels, 100, 100, 100)
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    out = F.pad(x, conv_padding, 'constant', pad_value)

    out = conv(out)
    print(out.shape): torch.Size([8, 16, 100, 100, 100])
    """
    def __init__(self, depth, height, width, pad_value):
        super(conv3d_padding_same, self).__init__()
        self.kernel_size = (depth, height, width)
        self.pad_value = pad_value

    def forward(self, x):
        # Determine amount of padding
        # Internal parameters used to reproduce Tensorflow "Same" padding.
        # For some reasons, padding dimensions are reversed wrt kernel sizes.
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        x_padded = torch.nn.functional.pad(x, conv_padding, 'constant', self.pad_value)

        return x_padded


class reshape_tensor(torch.nn.Module):
    """
    Reshape tensor.
    """
    def __init__(self, *args):
        super(reshape_tensor, self).__init__()
        self.output_dim = []
        for a in args:
            self.output_dim.append(a)

    def forward(self, x, batch_size):
        output_dim = [batch_size] + self.output_dim
        x = x.view(output_dim)

        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding='same', bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.norm1 = torch.nn.InstanceNorm3d(planes)
        self.activation = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3x3(planes, planes * self.expansion)
        self.norm2 = torch.nn.InstanceNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class InvertedResidual(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        interm_features = planes * self.expansion

        self.conv1 = conv1x1x1(in_planes, interm_features)
        self.norm1 = torch.nn.InstanceNorm3d(interm_features)
        self.conv2 = conv3x3x3(interm_features, interm_features, stride)
        self.norm2 = torch.nn.InstanceNorm3d(interm_features)
        self.conv3 = conv1x1x1(interm_features, planes)
        self.norm3 = torch.nn.InstanceNorm3d(planes)
        self.activation = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out
        
        
class ConvNextV2Block(torch.nn.Module):
    """ 
    ConvNeXtV2 Block.
    """
    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, stride=1, drop_path=0.):
        super().__init__()
        self.dwconv = torch.nn.Conv3d(in_planes, planes, kernel_size=7, stride=stride, padding=3, groups=in_planes) # depthwise conv
        self.norm = LayerNorm(planes, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(planes, 4 * planes) # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.grn = GRN(4 * planes)
        self.pwconv2 = torch.nn.Linear(4 * planes, planes)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 4, 1) # (B, C, T, H, W) -> (B, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 4, 1, 2, 3) # (B, T, H, W, C) -> (B, C, T, H, W)

        x = inp + self.drop_path(x)
        return x


class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_time_dim, pad_value, lrelu_alpha, use_activation,
                 use_bias=False):
        super(conv_block, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)
        
        if not pad_time_dim:
            kernel_depth = 1
        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)

        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.use_activation = use_activation
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        if self.use_activation:
            x = self.activation1(x)
        return x


class pooling_conv(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_time_dim, pad_value, lrelu_alpha, use_bias=False):
        super(pooling_conv, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        if not pad_time_dim:
            kernel_depth = 1
        self.pad = conv3d_padding_same(depth=kernel_depth, height=1, width=1, pad_value=pad_value)

        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.activation1(x)

        return x


class ResNet_DCRNN_LReLU(torch.nn.Module):
    """
    ResNet + DCNN + LSTM
    """
    def __init__(self, n_input_channels, depth, height, width, n_features, num_classes, block_name, filters, kernel_sizes, strides,
                 lstm_num_layers, lstm_hidden_size, lstm_dropout_p, lstm_bidirectional, pad_value, pad_time_dim, lrelu_alpha, 
                 dropout_p, pooling_conv_filters, perform_pooling, linear_units, use_bias, use_activation):
        super(ResNet_DCRNN_LReLU, self).__init__()
        self.n_features = n_features
        self.lstm_num_layers = lstm_num_layers
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling
        
        # Determine number of downsampling blocks
        n_down_blocks = [sum([x[0] == 2 for x in strides]), sum([x[1] == 2 for x in strides]), sum([x[2] == 2 for x in strides])]
        
        # Determine Linear input channel size
        # When using conv2d_padding_same(), the time-dimension has no padding, so end_depth is reduced by 
        # kernel_sizes[0][0] (i.e., kernel_size_time_dim) after each layer
        if pad_time_dim:
            end_depth = math.ceil(depth / (2 ** n_down_blocks[0])) if n_down_blocks[0] > 0 else depth
        else:
            end_depth = depth
            for i in range(len(filters)):
                end_depth -= (kernel_sizes[0][0] - 1)
                if i < n_down_blocks[0]:
                    end_depth = math.ceil(end_depth / 2)
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))
        
        # Initialize blocks
        if block_name == 'BasicResBlock':
            main_block = BasicResBlock
        elif block_name == 'InvertedResidual':
            main_block = InvertedResidual
        elif block_name == 'ConvNextV2Block':
            main_block = ConvNextV2Block
        else:
            raise ValueError('Block_name = {} is not available'.format(block_name))
            
        in_channels = [n_input_channels] + list(filters[:-1])
        self.blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):

            # Main block
            self.blocks.add_module('{}_{}'.format(block_name, i), 
                                   main_block(in_planes=in_channels[i], planes=in_channels[i],
                                              pad_value=pad_value, lrelu_alpha=lrelu_alpha))
            # Downsampling conv block
            self.blocks.add_module('Downsampling_conv_block_{}'.format(i),
                                   conv_block(in_channels=in_channels[i], filters=filters[i],
                                              kernel_size=kernel_sizes[i], strides=strides[i], 
                                              pad_time_dim=pad_time_dim, pad_value=pad_value,
                                              lrelu_alpha=lrelu_alpha, use_bias=use_bias,
                                              use_activation=use_activation))

        # Initialize pooling conv
        lstm_input_size = filters[-1]
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [1, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1, 
                                     pad_time_dim=pad_time_dim, pad_value=pad_value,
                                     lrelu_alpha=lrelu_alpha, use_bias=use_bias)
            end_depth, end_height, end_width = end_depth, 1, 1
            lstm_input_size = self.pooling_conv_filters
        elif self.perform_pooling:
            self.pool = torch.nn.AdaptiveMaxPool3d(output_size=(end_depth, 1, 1))
            end_depth, end_height, end_width = end_depth, 1, 1  # e.g.: 25, 1, 1

        # LSTM layers
        if self.lstm_num_layers > 0:
            self.lstm = torch.nn.LSTM(input_size=lstm_input_size * end_height * end_width,
                                      hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, 
                                      dropout=lstm_dropout_p, bidirectional=lstm_bidirectional, 
                                      batch_first=True)
            linear_in_features = lstm_hidden_size
            if lstm_bidirectional:
                linear_in_features = linear_in_features * 2
        else:
            linear_in_features = lstm_input_size * end_depth * end_height * end_width

        # Initialize flatten layer
        self.flatten = torch.nn.Flatten()

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [linear_in_features] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i + 1],
                                                          bias=use_bias))
            self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))

        # Initialize output layer
        self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes, bias=use_bias)

    def forward(self, x, features):
        # Blocks
        for block in self.blocks:
            x = block(x)

        # Pooling layers
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)

        if self.lstm_num_layers > 0:
            # LSTM
            # Swap hidden size (C, dim=1) and time dim (T, dim=2)
            x = torch.swapaxes(x, 1, 2)  # x.shape = (B, T, C, H, W)

            # Keep the first 2 dimension the same, and flatten hidden units at height and width:
            x = x.flatten(start_dim=2, end_dim=-1)  # x.shape = (B, T, C * height * width)

            x, (hn, cn) = self.lstm(x)  # x.shape = (B, T, H_out)

            # Decode the output of the last time step
            x = x[:, -1, :]  # x.shape = (B, H_out)
        else:
            x = self.flatten(x)  # x.shape = (B, H_out)

        # Linear layers
        for layer in self.linear_layers:
            x = layer(x)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        # Output
        x = x.float()
        x = self.out_layer(x)

        return x


