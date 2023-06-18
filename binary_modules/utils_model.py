import torch
import torch.nn as nn
from binary_modules.utils_quant import SoftmaxBinaryQuantizer, BinaryQuantizer, ZMeanBinaryQuantizer, BinaryLinear_STE, BinaryLinear_adapscaling, BinaryLinear_adapscaling_1w32a, BinaryLinear_STE_32a

class BiAttention_baseline(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = BinaryLinear_STE(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = BinaryLinear_STE(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.quant_layer = BinaryQuantizer().apply
        self.attn_quant_layer = ZMeanBinaryQuantizer().apply

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        binary_qkv = self.quant_layer(qkv)
        q, k, v = binary_qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.attn_quant_layer(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class BiAttention_SAB(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = BinaryLinear_adapscaling(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = BinaryLinear_adapscaling(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.quant_layer = BinaryQuantizer().apply
        self.attn_quant_layer = SoftmaxBinaryQuantizer().apply

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        binary_qkv = self.quant_layer(qkv)
        q, k, v = binary_qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = self.attn_quant_layer(attn)
        attn = self.attn_quant_layer(attn).float().detach() - attn.softmax(-1).detach() + attn.softmax(-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class BiMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop

        self.fc1 = BinaryLinear_STE(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = BinaryLinear_STE(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs)
        # self.bbc = MetaConv_v2()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class BiMlp_1w32a(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop

        self.fc1 = BinaryLinear_STE_32a(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = BinaryLinear_STE_32a(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class BiMlp_adapscaling_1w32a(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop

        self.fc1 = BinaryLinear_adapscaling_1w32a(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = BinaryLinear_adapscaling_1w32a(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
