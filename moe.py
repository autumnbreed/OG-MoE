import math
import numpy as np
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from einops import rearrange, repeat

import torch
import torch.nn.functional as F

class simple_cosine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha_ = config.cons_alpha
        self.temperature = 0.78
        
    def forward(self, A, B):
        # shape (B, max_patch)
        cos_sim = F.cosine_similarity(A, B, dim=-1)
        
        # temperature control
        scaled_cos_sim = cos_sim / (1 + self.temperature)
    
        # mean
        avg_scaled_cos_sim = scaled_cos_sim.mean()
        
        return self.alpha_ * avg_scaled_cos_sim


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class GeneMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.gating_temp = config.GE_temprature # < 1 supposed to be

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.gene_hide_dim = config.gene_size
        # self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.weight_token = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.weight_gene = nn.Parameter(torch.empty((self.n_routed_experts, self.gene_hide_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight_token, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_gene, a=math.sqrt(5))
    
    def forward(self, hidden_states, gene_vectors):
        bsz, seq_len, h = hidden_states.shape
        _, gene_len = gene_vectors.shape
        # bsz, gene_len = gene_vectors.shape      
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        gene_bias = repeat(gene_vectors, 'b g -> b n g', n=seq_len)
        #gene_bias = gene_bias.view(-1, gene_len)
        gene_bias = rearrange(gene_bias, 'b n g -> (b n) g', g=gene_len)
        
        logits_h = F.linear(hidden_states, self.weight_token, None)
        logits_g = F.linear(gene_bias, self.weight_gene, None)
        
        logits = (logits_h + logits_g / self.gating_temp) / (1 + 1/self.gating_temp)
        
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        #if self.training and self.alpha > 0.0:
        if self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class DMLP(nn.Module):
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.LeakyReLU() # ACT2FN[config.hidden_act]

    def forward(self, x):
        # if self.config.pretraining_tp > 1:
        #     slice = self.intermediate_size // self.config.pretraining_tp
        #     gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        #     up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        #     down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        #     gate_proj = torch.cat(
        #         [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        #     )
        #     up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        #     intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        #     down_proj = [
        #         F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        #     ]
        #     down_proj = sum(down_proj)
        # else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    

class MyMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([DMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = GeneMoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DMLP(config=config, intermediate_size = intermediate_size)
            
        self.cos = simple_cosine(config)
    
    def forward(self, hidden_states, g=None, attn_mask=None):
        identity = hidden_states
        orig_shape = hidden_states.shape
        if g is not None:
            g = g.view(-1, g.shape[-1])  # (bsz, seq_len, gene_len) -> (bsz*seq_len, gene_len)
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states, g)
        else:
            g = torch.ones(self.config.gene_size)
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states, g)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        # if self.training:
        hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y =  y.view(*orig_shape)
        y = AddAuxiliaryLoss.apply(y, aux_loss)
        
        # if self.config.n_shared_experts is not None:
        p = self.shared_experts(identity)
        y = y + p
        
        # Cosine loss head MoE MLP vs Shared MLP
        cLoss = self.cos(y, p) 
        y = AddAuxiliaryLoss.apply(y, cLoss)
            
        return y

if __name__ == "__main__":
    bsize, max_patches, patch_dim = 8, 158, 128
    patches = torch.tensor(np.random.randn(bsize, max_patches, patch_dim).astype(np.float32))
    gene_patches = torch.tensor(np.random.randn(bsize, 126*2).astype(np.float32))
    
    from model import DemoConfig
    import json
    with open("./config.json", "r") as f:
        config_dict = json.load(f)

    config = DemoConfig(**config_dict)
    
    test_moe = MyMoE(config)
    
    y = test_moe(patches, gene_patches)
    


 
    # @torch.no_grad()
    # def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    #     expert_cache = torch.zeros_like(x)
    #     idxs = flat_expert_indices.argsort()
    #     tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    #     token_idxs = idxs // self.num_experts_per_tok
    #     for i, end_idx in enumerate(tokens_per_expert):
    #         start_idx = 0 if i == 0 else tokens_per_expert[i-1]
    #         if start_idx == end_idx:
    #             continue
    #         expert = self.experts[i]
    #         exp_token_idx = token_idxs[start_idx:end_idx]
    #         expert_tokens = x[exp_token_idx]
    #         expert_out = expert(expert_tokens)
    #         expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
    #         expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
    #     return expert_cache

