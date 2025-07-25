"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from Codes import sign_to_bin, bin_to_sign
import numpy as np

###################################################
###################################################

def diff_syndrome(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    # x_bin = sign_to_bin(x) if -1 in x else x
    x_bin = x

    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    # assert torch.allclose(logical_flipped(H_bin,x_bin).cpu().detach().bool(), tmp.detach().cpu().bool())
    return tmp

def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2

###################################################
###################################################
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # scores = oe.contract("x h n d, x h m d, a a n m -> x h n m", query,key,~mask)/ math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

############################################################


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        self.no_g = args.no_g
        code = args.code
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, dropout)

        self.src_embed = torch.nn.Parameter(torch.empty(
            (code.n + code.pc_matrix.size(0), args.d_model)))
        self.decoder = Encoder(EncoderLayer(
            args.d_model, c(attn), c(ff), dropout), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)
        #
        N_in = 5
        non_lin_fun = torch.nn.GELU
        layers = [torch.nn.Linear(code.pc_matrix.size(0), N_in*code.n), non_lin_fun()]
        for _ in range(1):
            layers += [torch.nn.Linear(N_in*code.n, N_in*code.n),non_lin_fun()]
        layers += [torch.nn.Linear(N_in*code.n, code.n)]
        #
        self.syn_to_noise = torch.nn.Sequential(*layers)
        #
        self.get_mask(code)
        if args.no_mask > 0:
            self.src_mask = None
        logging.info(f'Mask:\n {self.src_mask}')
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.magnitude_pred = []

    def forward(self, magnitude, syndrome):
        magnitude = self.syn_to_noise(syndrome)
        if self.no_g:
            magnitude = magnitude*0+1
        self.magnitude_pred.append(magnitude)
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2):
        #dealing with DP activations
        indices = [xx.device.index for xx in self.magnitude_pred]
        if len(list(set(indices))) > 1:
            device_zero = self.magnitude_pred[np.where(np.array(indices)==0)[0][0]]
            self.magnitude_pred = torch.cat([self.magnitude_pred[np.where(np.array(indices)==ii)[0][0]].to(device_zero) for ii in range(len(self.magnitude_pred))],0)
        else:
            self.magnitude_pred = self.magnitude_pred[0]
        loss1 = F.binary_cross_entropy_with_logits(z_pred, 1-z2)
        loss2 = F.binary_cross_entropy_with_logits(self.magnitude_pred, 1-z2)
        ###
        self.magnitude_pred = []
        return loss1,loss2

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        ###
        src_mask = build_mask(code)
        mask_size = code.n + code.pc_matrix.size(0)
        a = mask_size ** 2
        logging.info(
            f'Self-Attention Sparsity Ratio={100 * torch.sum((src_mask).float()) / a:0.2f}%, Self-Attention Complexity Ratio={100 * torch.sum((~src_mask).float())/2 / a:0.2f}%')
        self.register_buffer('src_mask', src_mask)


############################################################
############################################################

if __name__ == '__main__':
    pass
