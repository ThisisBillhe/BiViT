from distutils.log import error
from http.client import UnimplementedFileMode
from select import select
from unicodedata import numeric
import torch
import torch.nn as nn
# import pdb
# import matplotlib.pyplot as plt
import math
# from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import custom_fwd, custom_bwd


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

class SoftmaxBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        attn = input.softmax(dim=-1)
        thresh, idx = torch.max(attn, dim=-1) 
        thresh *= 0.25 
        thresh = thresh.unsqueeze(-1)
        out = torch.where(attn < thresh, 0, 1)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)

class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class OnehotBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        idx=torch.argmax(input, dim=-1)
        out = F.one_hot(idx, input.shape[-1])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

class ProbBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        attn = input.softmax(-1)
        out = torch.where(torch.rand_like(attn) < attn, 1, 0)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

class ZMeanBinaryQuantizer_withScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scaling_factor = torch.sum(F.relu(input))
        n = torch.count_nonzero(F.relu(input))
        # print(input.shape)
        # print(scaling_factor.shape)
        # print(n.shape)
        scaling_factor = (scaling_factor / n).detach()

        out = torch.sign(input)
        out[out==-1] = 0
        out = out * scaling_factor

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None



class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None
  

class BinaryLinear_STE(nn.Linear):
    def __init__(self,  *kargs, bias=True, quantize_act=True, weight_bits=1, input_bits=1, clip_val=2.5):
        super(BinaryLinear_STE, self).__init__(*kargs,bias=True)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits

        # self.weight_quantizer = BinaryQuantizer
        # self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        self.init = True
        # nn.init.uniform_(self.weight, -0.015, 0.015)
        if self.quantize_act:
            self.input_bits = input_bits
            # self.act_quantizer = BinaryQuantizer
            self.act_quant_layer = BinaryQuantizer().apply
            self.weight_quant_layer = BinaryQuantizer().apply
            # self.weight_quant_layer = MetaConv_v2()
            # self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
    #     self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))
 
    # def reset_scale(self, input):
    #     bw = self.weight
    #     ba = input
    #     self.scale = Parameter((ba.norm() / torch.sign(ba).norm()).float().to(ba.device))

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            weight = scaling_factor * self.weight_quant_layer(real_weights)
            # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            # weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        if self.input_bits == 1:
            ba = self.act_quant_layer(input)
            # binary_input_no_grad = torch.sign(input)
            # cliped_input = torch.clamp(input, -1.0, 1.0)
            # ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        
        out = F.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out


class BinaryLinear_STE_32a(nn.Linear):
    def __init__(self,  *kargs, bias=True, quantize_act=True, weight_bits=1, input_bits=1, clip_val=2.5):
        super(BinaryLinear_STE_32a, self).__init__(*kargs,bias=True)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits

        # self.weight_quantizer = BinaryQuantizer
        # self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        self.init = True
        # nn.init.uniform_(self.weight, -0.015, 0.015)
        if self.quantize_act:
            self.input_bits = input_bits
            # self.act_quantizer = BinaryQuantizer
            self.weight_quant_layer = BinaryQuantizer().apply
            # self.weight_quant_layer = MetaConv_v2()
            # self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
    #     self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))
 
    # def reset_scale(self, input):
    #     bw = self.weight
    #     ba = input
    #     self.scale = Parameter((ba.norm() / torch.sign(ba).norm()).float().to(ba.device))

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            weight = scaling_factor * self.weight_quant_layer(real_weights)
            # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            # weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        out = F.linear(input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out

class BinaryLinear_adapscaling(nn.Linear):
    def __init__(self,  *kargs, bias=True, quantize_act=True, weight_bits=1, input_bits=1, clip_val=2.5):
        super(BinaryLinear_adapscaling, self).__init__(*kargs,bias=True)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits

        self.init = True
        nn.init.uniform_(self.weight, -0.015, 0.015)
        if self.quantize_act:
            self.input_bits = input_bits
            self.act_quant_layer = BinaryQuantizer().apply
            self.weight_quant_layer = BinaryQuantizer().apply

        self.scaling_factor = nn.Parameter(
            torch.zeros((self.out_features, 1)), requires_grad=True
        )
        self.first_run = True

    def forward(self, input):
        if self.first_run:
            self.first_run = False
            if torch.sum(self.scaling_factor.data) == 0:
                print('attn initing scaling_factor...')
                self.scaling_factor.data = torch.mean(abs(self.weight), dim=1, keepdim=True)

        if self.weight_bits == 1:
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            weight = self.scaling_factor * self.weight_quant_layer(real_weights)
            # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            # weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        if self.input_bits == 1:
            ba = self.act_quant_layer(input)
            # binary_input_no_grad = torch.sign(input)
            # cliped_input = torch.clamp(input, -1.0, 1.0)
            # ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        
        out = F.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out


class BinaryLinear_32w1a(nn.Linear):
    def __init__(self,  *kargs, bias=True, quantize_act=True, weight_bits=1, input_bits=1, clip_val=2.5):
        super(BinaryLinear_32w1a, self).__init__(*kargs,bias=True)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits

        self.init = True
        nn.init.uniform_(self.weight, -0.015, 0.015)
        if self.quantize_act:
            self.input_bits = input_bits
            self.act_quant_layer = BinaryQuantizer().apply

    def forward(self, input):


        if self.input_bits == 1:
            ba = self.act_quant_layer(input)
            # binary_input_no_grad = torch.sign(input)
            # cliped_input = torch.clamp(input, -1.0, 1.0)
            # ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        
        out = F.linear(ba, self.weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out

class BinaryLinear_adapscaling_1w32a(nn.Linear):
    def __init__(self,  *kargs, bias=True, quantize_act=True, weight_bits=1, input_bits=1, clip_val=2.5):
        super(BinaryLinear_adapscaling_1w32a, self).__init__(*kargs,bias=True)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits

        self.init = True
        nn.init.uniform_(self.weight, -0.015, 0.015)
        if self.quantize_act:
            self.input_bits = input_bits
            self.weight_quant_layer = BinaryQuantizer().apply

        self.scaling_factor = nn.Parameter(
            torch.zeros((self.out_features, 1)), requires_grad=True
        )
        self.first_run = True

    def forward(self, input):
        if self.first_run:
            self.first_run = False
            if torch.sum(self.scaling_factor.data) == 0:
                print('mlp initing scaling_factor...')
                self.scaling_factor.data = torch.mean(abs(self.weight), dim=1, keepdim=True)

        if self.weight_bits == 1:
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            weight = self.scaling_factor * self.weight_quant_layer(real_weights)
            # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            # weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        
        out = F.linear(input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out


class ScaleLayer(nn.Module):

    def __init__(self, num_features, use_bias=False, scale_init=1.0):
        super(ScaleLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features),requires_grad=True)
        nn.init.constant_(self.weight, scale_init)
        self.num_features = num_features
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(num_features),requires_grad=True)
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, 1, self.num_features)
        else:
            return inputs * self.weight.view(1, 1, self.num_features) + self.bias.view(1, 1, self.num_features)