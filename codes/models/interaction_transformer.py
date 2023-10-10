"""
Copyright 2023 LINE Corporation

LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip

import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalInteractionCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross = True
        if self.cross:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x1, x2, emb, src_mask, timesteps):
        """
        x1, x2: B, T, D
        x1 -> x2
        """
        if self.cross:
            x1, x2 = torch.cat([x1, x2], dim=0), torch.cat([x2, x1], dim=0)

        B, T, D = x1.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x1))
        # B, N, D
        key = (self.key(self.norm(x2)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        
        # B, T, H, HD
        value = self.value(self.norm(x2)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        if self.cross:
            y = x1 + self.proj_out(y, emb)
            return y[:B//2], y[B//2:]
        else:
            return x1 + y, x2

class TemporalInteractionCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross = True
        if self.cross:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x1, x2, emb, src_mask):
        """
        x1, x2: B, T, D
        """
        if self.cross:
            x1, x2 = torch.cat([x1, x2], dim=0), torch.cat([x2, x1], dim=0)
        B, T, D = x1.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x1)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.text_norm(x2)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + (1 - src_mask[:B]) * -100000

        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(x2)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        if self.cross:
            y = x1 + self.proj_out(y, emb)
            return y[:B//2], y[B//2:]
        else:
            return x1 + y, x2

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
    
class TemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + src_mask
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 no_cross_attn=False
                 ):
        super().__init__()
        self.no_cross_attn = no_cross_attn
        self.sa_block = LinearTemporalSelfAttention(
                seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearTemporalCrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        if not self.no_cross_attn:
            self.int_ca_block = LinearTemporalInteractionCrossAttention(
                    seq_len, latent_dim, latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
    
    def forward(self, x1, x2, xf, emb, src_mask, timesteps):
        batch_size = x1.size(0)
        x = torch.cat([x1, x2], dim=0)
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x1, x2 = x[:batch_size], x[batch_size:]
        if not self.no_cross_attn:
            x1, x2 = self.int_ca_block(x1, x2, emb, src_mask, timesteps)
        x = torch.cat([x1, x2], dim=0)
        x = self.ffn(x, emb)
        return x[:batch_size], x[batch_size:]

class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 no_cross_attn=False,
                 dropout=0.1):
        super().__init__()
        self.no_cross_attn = no_cross_attn
        self.sa_block = TemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        if not self.no_cross_attn:
            self.ca_block = TemporalCrossAttention(
                seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        if not self.no_cross_attn:
            x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x


class MotionInteractionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_text_layers=4,
                 text_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 no_eff=False,
                 no_cross_attn=False,
                 cap_id=False,
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.cap_id = cap_id

        # Text Transformer
        if self.cap_id:
            self.cap_embedding = nn.Parameter(torch.randn(43, text_latent_dim))
            self.text_proj = nn.Sequential(
                nn.Linear(text_latent_dim, self.time_embed_dim)
            )
        else:
            self.clip, _ = clip.load('ViT-B/32', "cpu")
            self.no_clip = no_clip
            if no_clip:
                self.clip.d_type = self.clip.visual.conv1.weight.dtype
                self.clip.initialize_parameters()
                del self.clip.visual
                del self.clip.logit_scale
                del self.clip.text_projection
            else:
                set_requires_grad(self.clip, False)
            if text_latent_dim != 512:
                self.text_pre_proj = nn.Linear(512, text_latent_dim)
            else:
                self.text_pre_proj = nn.Identity()
            textTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=text_latent_dim,
                nhead=text_num_heads,
                dim_feedforward=text_ff_size,
                dropout=dropout,
                activation=activation)
            self.textTransEncoder = nn.TransformerEncoder(
                textTransEncoderLayer,
                num_layers=num_text_layers)
            self.text_ln = nn.LayerNorm(text_latent_dim)
            self.text_proj = nn.Sequential(
                nn.Linear(text_latent_dim, self.time_embed_dim)
            )

        # Input Embedding
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.two_embed = True
        if self.two_embed:
            self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
            self.joint_embed2 = nn.Linear(4, self.latent_dim)
        else:
            self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
            self.init_pos_embedding = nn.Parameter(torch.randn(1, latent_dim))

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            if no_eff:
                self.temporal_decoder_blocks.append(
                    TemporalDiffusionTransformerDecoderLayer(
                        seq_len=num_frames,
                        latent_dim=latent_dim,
                        text_latent_dim=text_latent_dim,
                        time_embed_dim=self.time_embed_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout
                    )
                )
            else:
                self.temporal_decoder_blocks.append(
                    LinearTemporalDiffusionTransformerDecoderLayer(
                        seq_len=num_frames,
                        latent_dim=latent_dim,
                        text_latent_dim=text_latent_dim,
                        time_embed_dim=self.time_embed_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout,
                        no_cross_attn=no_cross_attn,
                    )
                )
        
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        self.out2 = zero_module(nn.Linear(self.latent_dim, self.input_feats))
    
    def load_my_state_dict(self, state_dict, opt):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if opt.only_language:
                if name not in own_state or ('clip' not in name and 'text' not in name):
                    print(name)
                    continue
            elif opt.only_motion:
                if name not in own_state or ('clip' in name or 'text' in name):
                    print(name)
                    continue
            else:
                if name not in own_state:
                    if not(opt.cap_id and ('clip' in name or 'text' in name)):
                        print(name)
                    continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def encode_text(self, text, device):
        if self.no_clip:
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.d_type)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.d_type)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.d_type)
        else:
            with torch.no_grad():
                text = clip.tokenize(text, truncate=True).to(device)
                x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

                x = x + self.clip.positional_embedding.type(self.clip.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip.transformer(x)
                x = self.clip.ln_final(x).type(self.clip.dtype)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def get_class_embedding(self, text):
        text = torch.cat(text)
        xf_proj = self.cap_embedding[text]
        xf_out = xf_proj.unsqueeze(1)
        xf_proj = self.text_proj(xf_proj)
        return xf_proj, xf_out

    def generate_src_mask(self, T, length):
        B = len(length)
        
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, x, timesteps, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        [x1, x2] : [x[:B//2], x[B//2:]]
        """
        B = x.shape[0]//2
        x1, x2 = x[:B], x[B:]
        T = x1.shape[1]
        if self.cap_id:
            xf_proj, xf_out = self.get_class_embedding(text)
        else:
            if xf_proj is None or xf_out is None:
                xf_proj, xf_out = self.encode_text(text, x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        def embed_motion(motion):
            if self.two_embed:
                move = self.joint_embed(motion[:, 1:]) + self.sequence_embedding.unsqueeze(0)[:, :T-1]
                init_pos = self.joint_embed2(motion[:, 0, :4])
            else:
                h = self.joint_embed(motion)
                move = h[:, :T-1, :] + self.sequence_embedding.unsqueeze(0)[:, :T-1, :]
                init_pos = h[:, -1, :] + self.init_pos_embedding
            embed = torch.cat([init_pos.view(B,1,-1), move],axis=1)
            return embed

        # B, T, latent_dim
        h1 = embed_motion(x1)
        h2 = embed_motion(x2)

        src_mask = self.generate_src_mask(T, length)
        src_mask = src_mask.to(x1.device).unsqueeze(-1)
        for module in self.temporal_decoder_blocks:
            h1, h2 = module(h1, h2, xf_out, emb, src_mask, timesteps)

        output1 = torch.cat([self.out2(h1[:,0]).view(B, 1, -1).contiguous(), self.out(h1[:,1:]).view(B, T-1, -1).contiguous()], axis=1)
        output2 = torch.cat([self.out2(h2[:,0]).view(B, 1, -1).contiguous(), self.out(h2[:,1:]).view(B, T-1, -1).contiguous()], axis=1)
        
        return torch.cat([output1, output2], dim=0)

class LinearTemporalBaselineDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = LinearTemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearTemporalCrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x

class MotionEncoder(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 class_num=26,
                 activation="gelu", 
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.init_pos_embedding = nn.Parameter(torch.randn(1, latent_dim))


        # Input Embedding
        self.joint_embed1 = nn.Linear(self.input_feats, self.latent_dim)
        self.joint_embed2 = nn.Linear(4, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        motionTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.motionTransEncoder = nn.TransformerEncoder(
            motionTransEncoderLayer,
            num_layers=num_layers
        )

        # Output Module
        self.out1 = zero_module(nn.Linear(latent_dim, latent_dim))
        self.out2 = zero_module(nn.Linear(latent_dim, latent_dim))

        self.fin_proj = nn.Sequential(
            nn.Linear(latent_dim, class_num)
        )

    def get_class_embedding(self, text):
        text = torch.cat(text)
        xf_proj = self.cap_embedding[text]
        xf_out = xf_proj.unsqueeze(1)
        xf_proj = self.text_proj(xf_proj)
        return xf_proj, xf_out

    def generate_src_mask(self, T, length):
        B = len(length)
        
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return torch.cat([src_mask, src_mask], dim=1)

    def forward(self, x1, x2, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        [x1, x2] : [x[:B//2], x[B//2:]]
        """
        B = x1.shape[0]
        T = x1.shape[1]
        # B, T, latent_dim

        def embed_motion(motion):
            move = self.joint_embed1(motion[:, 1:, :]) + self.sequence_embedding.unsqueeze(0)[:, :T-1, :]
            init_pos = self.joint_embed2(motion[:, 0, :4])
            return torch.cat([init_pos.view(B,1,-1), move],axis=1)

        h1 = embed_motion(x1)
        h2 = embed_motion(x2)

        src_mask = self.generate_src_mask(T, length).to(x1.device)

        motion = torch.cat([h1, h2], dim=1)
        motion_feature = self.motionTransEncoder(motion, src_key_padding_mask=(1-src_mask).type(torch.bool))
        h1, h2 = motion_feature[:,:T], motion_feature[:,T:]

        output1 = torch.cat([self.out2(h1[:,0]).view(B, 1, -1).contiguous(), self.out1(h1[:,1:]).view(B, T-1, -1).contiguous()], axis=1)
        output2 = torch.cat([self.out2(h2[:,0]).view(B, 1, -1).contiguous(), self.out1(h2[:,1:]).view(B, T-1, -1).contiguous()], axis=1)
        src_mask = src_mask.unsqueeze(-1)
        motion_feature = (torch.cat([output1, output2], dim=1)*src_mask).sum(dim=1)/src_mask.sum(dim=1)#.mean(dim=1)
        return self.fin_proj(motion_feature), motion_feature

class MotionConsistencyEvalModel(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 interaction_class_num=26,
                 class_num=2,
                 activation="gelu", 
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.init_pos_embedding = nn.Parameter(torch.randn(1, latent_dim))
        self.cls_input = nn.Parameter(torch.randn(1, 1, latent_dim))


        # Input Embedding
        self.joint_embed1 = nn.Linear(self.input_feats, self.latent_dim)
        self.joint_embed2 = nn.Linear(4, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        motionTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.motionTransEncoder = nn.TransformerEncoder(
            motionTransEncoderLayer,
            num_layers=num_layers
        )

        self.cls_output = nn.Sequential(
            nn.Linear(latent_dim, class_num)
        )

    def generate_src_mask(self, T, length):
        B = len(length)
        
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return torch.cat([torch.ones(B,1), src_mask, src_mask], dim=1)

    def forward(self, x1, x2, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        [x1, x2] : [x[:B//2], x[B//2:]]
        """
        B = x1.shape[0]
        T = x1.shape[1]
        # B, T, latent_dim

        def embed_motion(motion):
            move = self.joint_embed1(motion[:, 1:, :]) + self.sequence_embedding.unsqueeze(0)[:, :T-1, :]
            init_pos = self.joint_embed2(motion[:, 0, :4])
            return torch.cat([init_pos.view(B,1,-1), move],axis=1)

        h1 = embed_motion(x1)
        h2 = embed_motion(x2)

        src_mask = self.generate_src_mask(T, length).to(x1.device)

        motion = torch.cat([self.cls_input.repeat(B,1,1), h1, h2], dim=1)
        motion_feature = self.motionTransEncoder(motion, src_key_padding_mask=(1-src_mask).type(torch.bool))
        return self.cls_output(motion_feature[:,0])


if __name__=='__main__':
    '''
    for test
    '''
    def build_models(dim_pose=263):
        encoder = MotionInteractionTransformer(
            input_feats=dim_pose,
            num_frames=196,
            num_layers=8,
            latent_dim=512,
            no_clip=False,
            no_eff=False,
            cap_id=False
            )
        return encoder

    encoder = build_models().cuda()
    encoder.load_state_dict(torch.load('checkpoints/ntu_mul/interaction/model/latest.tar')['encoder'])
    x1 = torch.rand(1, 32, 263).cuda()
    x2 = torch.rand(1, 32, 263).cuda()
    timesteps = torch.tensor([10]).cuda()
    length = 31
    print(encoder(torch.cat([x1, x2]), timesteps, text=['a man', 'a woman'], length=[length,length]))
    print(encoder(torch.cat([x1[:,:length], x2[:,:length]]), timesteps, text=['a man', 'a woman'], length=[length,length]))


