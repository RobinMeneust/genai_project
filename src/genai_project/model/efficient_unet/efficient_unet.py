# https://arxiv.org/pdf/2205.11487  page 43

import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, channels, group_norm_groups=32):
        super(ResNetBlock, self).__init__()

        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.group_norm_1 = nn.GroupNorm(group_norm_groups, channels)
        self.swish_1 = nn.SiLU()

        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.group_norm_2 = nn.GroupNorm(group_norm_groups, channels)
        self.swish_2 = nn.SiLU()

        self.conv_3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        res = x
        out = self.conv_1(x)
        out = self.group_norm_1(out)
        out = self.swish_1(out)

        out = self.conv_2(out)
        out = self.group_norm_2(out)
        out = self.swish_2(out)

        out += self.conv_3(res)  # Skip

        return out

class DBlock(nn.Module):
    def __init__(self, channels, num_resnet_blocks_per_block, stride, group_norm_groups=32, use_conv=True, use_self_attention=True):
        super(DBlock, self).__init__()

        self.use_conv = use_conv
        self.use_self_attention = use_self_attention

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)

        res_blocks = []
        for _ in range(num_resnet_blocks_per_block):
            res_blocks.append(ResNetBlock(channels, group_norm_groups))
        self.res_blocks = nn.ModuleList(res_blocks)

        if use_self_attention:
            hidden_size = 2 * channels  # Hidden size is 2 × channels
            # we have to expand dims because in the paper they use hidden size of 2 * channels and out = channels
            self.expand_dim = nn.Conv2d(channels, hidden_size, kernel_size=1)
            self.attention = nn.MultiheadAttention(embed_dim=2*channels, num_heads=8)
            self.reduce_dim = nn.Conv2d(hidden_size, channels, kernel_size=1)

    def forward(self, x, conditional_embs, full_contextual_text_embs=None):
        # TODO: what is full_contextual_text_embs in the paper?
        if self.use_conv:
            x = self.conv(x)

        # combine conditional embeddings and x
        x += conditional_embs

        for res_block in self.res_blocks:
            x = res_block(x)

        if self.use_self_attention:
            x = self.expand_dim(x)
            x = self.attention(x)
            x = self.reduce_dim(x)
        
        return x
    
class UBlock(nn.Module):
    def __init__(self, channels, num_resnet_blocks_per_block, stride, group_norm_groups=32, use_self_attention=True, use_conv=True):
        super(UBlock, self).__init__()

        self.use_conv = use_conv
        self.use_self_attention = use_self_attention

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)

        res_blocks = []
        for _ in range(num_resnet_blocks_per_block):
            res_blocks.append(ResNetBlock(channels, group_norm_groups))
        self.res_blocks = nn.ModuleList(res_blocks)

        if use_self_attention:
            self.attention = nn.MultiheadAttention(embed_dim=2*channels, num_heads=8) # params not mentioned in the paper
        
    
    def forward(self, x, skip_x, conditional_embs=None):
        x = x + skip_x  # Skip connection with dblock

        for res_block in self.res_blocks:
            x = res_block(x)

        if self.use_self_attention:
            x = self.attention(x)

        if self.use_conv:
            x = self.conv(x)
        
        return x
        

class EfficientUnet(nn.Module):
    def __init__(self, in_channels=128, num_dblocks=5, in_size=64, out_size=256):
        # # num_dblocks = num_ublocks ?
        # super(EfficientUnet, self).__init__()

        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # self.dblocks = []
        # for i in range(num_dblocks):
        #     dblock = DBlock(in_channels, num_resnet_blocks_per_block=3, stride=2)
        #     self.dblocks.append(dblock)
        #     in_channels *= 2

        # # 256 dblocks
        pass
        