import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import numpy as np
from utils import evaluate_multiclass_metrics
from collections import Counter

from moe import MyMoE, DMLP, AddAuxiliaryLoss, GeneMoEGate, MoEGate

class DemoConfig(PretrainedConfig):
    model_type = "MiniViT"
    def __init__(
        self, 
        hidden_size=128, 
        num_labels1=2,
        num_labels0=3,
        num_moe_layers=8,
        num_experts_per_tok=4,
        n_routed_experts=16,
        moe_intermediate_size=32,
        cons_alpha=0.05,
        aux_loss_alpha=0.1,
        seq_aux=False,
        scoring_func="softmax",
        norm_topk_prob=True,
        g_size=104,
        GE_temprature=0.7,
        num_mlp_layers=2,
        intermediate_size=256,
        n_shared_experts=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels0 = num_labels0
        self.num_labels1 = num_labels1
        self.num_moe_layers = num_moe_layers
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.cons_alpha = cons_alpha
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.g_size = g_size
        self.GE_temprature = GE_temprature
        self.num_mlp_layers = num_mlp_layers
        self.intermediate_size = intermediate_size
        self.n_shared_experts = n_shared_experts
        
# ------------------ Mask Gen ------------------
def att_mask(seq_len=None):
    if not seq_len:
        seq_len = 158
    # [seq_len, seq_len]
    attn_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)

    # attention mask format
    attn_mask[2:80, 80:158] = True  
    attn_mask[80:158, 2:80] = True  

    #print(attn_mask.shape)

    return attn_mask


def token_mask(x, mask_rate=0.15):
    """
    x: [batch_size, seq_len, hidden_size]
    mask_rate: 0.15 defaults
    """
    batch_size, seq_len, hidden_size = x.shape
    y = x.clone()
    num_mask = int(seq_len * mask_rate)

    for b in range(batch_size):
        perm = torch.randperm(seq_len)
        mask_indices = perm[:num_mask]
        y[b, mask_indices, :] = 0.0

    return y, mask_indices

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768, image_size=224):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

    def forward(self, x):
        x = self.proj(x)            # [B, emb_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        return x

class PatchEmbeddingWithCLS(nn.Module):
    def __init__(self, 
                 input_dim=360,        # token
                 seq_len=156,          # 
                 emb_dim=128,          # embedding
                 pos_len=78,           #
                 ):
        """
        """
        super().__init__()
        self.seq_len = seq_len  # 156
        self.half = seq_len // 2  # 78
        
        if input_dim != emb_dim:
            self.proj = nn.Linear(input_dim, emb_dim)
        else:
            self.proj = nn.Identity()
        
        self.emb_dim = emb_dim
        
        # CLS token: cls0 , cls1, [1, 1, emb_dim]
        self.cls0 = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.cls1 = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        self.pos_embed = nn.Parameter(torch.rand(1, pos_len, emb_dim))
        
        # self.dropout = nn.Dropout(p=dropout)
        nn.init.normal_(self.cls0, std=0.02)
        nn.init.normal_(self.cls1, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.size(0)
        # [B, 156, emb_dim]
        x = self.proj(x)
        
        # 2. Pos：self.pos_embed: [1, 78, emb_dim]
        pos_embed_full = torch.cat([self.pos_embed, self.pos_embed], dim=1)
        
        x = x + pos_embed_full  # [B, 156, emb_dim]
        
        cls0 = self.cls0.expand(B, -1, -1)  # [B, 1, emb_dim]
        cls1 = self.cls1.expand(B, -1, -1)  # [B, 1, emb_dim]

        x = torch.cat([cls0, cls1, x], dim=1)  # [B, 2+156, emb_dim] = [B, 158, emb_dim]
        return x
    
class MinimalTransformerATT(nn.Module):
    """
    Transformer Block + FFN
    """
    def __init__(self, hidden_size, num_heads, dropout_rito, activation="gelu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rito, bias=False, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)

    def forward(self, x, g=0, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        return x
    
class MinimalTransformerFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rito):
        super().__init__()
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout_rito)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x, g=0, attn_mask=None):
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        ffn_out = self.norm2(x + h)
        return ffn_out

class SimpleGeneEmbedding(nn.Module):
    def __init__(self, input_dim=324, output_dim=104):
        super(SimpleGeneEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        return embedding


class MiniViT(PreTrainedModel):
    """
    """
    config_class = DemoConfig
    
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        #self.num_layers = config.num_hidden_layers
        self.drop_rate = config.dropout

        # Embedding
        # self.patch_embed = nn.Linear(config.patch_dim, config.hidden_size)
        self.patch_embed = PatchEmbeddingWithCLS(
                input_dim=config.patch_dim,
                seq_len=config.max_patches,
                emb_dim=self.hidden_size,
                pos_len=78,)
        
        self.Gene_embed = SimpleGeneEmbedding(input_dim=324, output_dim=config.gene_size)

        # 多层 transformer block
        blocks_seq = []
        for _ in range(config.num_mlp_layers):
            blocks_seq.extend(
                [MinimalTransformerATT(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_rito=self.drop_rate,
                activation="gelu",
                
            ),
                MinimalTransformerFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout_rito=self.drop_rate,
            )]
            )
            
        for _ in range(config.num_moe_layers):
            blocks_seq.extend(
                [MinimalTransformerATT(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_rito=self.drop_rate,
                activation="gelu",
                
            ),
                MyMoE(config)]
            )
            
        self.blocks = nn.ModuleList(blocks_seq)

        # label1, label2
        self.cls_head1 = nn.Sequential(nn.Linear(config.hidden_size, 100), nn.BatchNorm1d(100, affine=False), nn.Linear(100, config.num_labels0))
        self.cls_head2 = nn.Sequential(nn.Linear(config.hidden_size, 100), nn.BatchNorm1d(100, affine=False), nn.Linear(100, config.num_labels1))
        # reconstruction mask patch (MSE)
        self.mse_head = nn.Linear(config.hidden_size, config.patch_dim)
        
        self.post_init()

        # embedding hook
        self.emb_hook_output = None
        # hook output
        def hook_fn(module, input_, output_):
            self.emb_hook_output = output_.detach()

        self.blocks[-1].register_forward_hook(hook_fn)

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        patch_tensors,    # [B, seq_len, 128]
        gene_tensors,
        label0=None, 
        label1=None,
        original_patch=None,
        mask_idx=None,
        criterion = nn.CrossEntropyLoss(),
        **kwargs
    ):
        """
        - warm_start=True:  MSE Loss
        - warm_start=False: Loss + MSE Loss
        SequenceClassifierOutputWithPast
        """
        device = patch_tensors.device
        B, seq_len, _ = patch_tensors.shape
        B2, _ = gene_tensors.shape
        assert B == B2
        
        # 1) Project layer hidden
        hidden_states = self.patch_embed(patch_tensors)  # [B, seq_len, hidden_size]
        
        g_vec = self.Gene_embed(gene_tensors)

        # 2) Attention mask

        
        # 3) mask 15% token (not 2 CLS)

        
        # 4) transformer
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, g_vec, attn_mask=attn_mask)

        # 5) CLS output
        cls_vec1 = hidden_states[:, 0, :]  # [B, hidden_size]
        cls_vec2 = hidden_states[:, 1, :]  # [B, hidden_size]
        logits0 = self.cls_head1(cls_vec1) # [B, num_labels1]
        logits1 = self.cls_head2(cls_vec2) # [B, num_labels2]
        
        # 5) MSE on masked patches
        reconstruct_loss = 0.0
        # if mask_idx.numel() == 0:
        #     reconstruct_loss += 0.0
        if (original_patch is not None) and (mask_idx is not None):
            masked_hidden = hidden_states[:, mask_idx, :]  # [B, M, D]
            B, M, D = masked_hidden.shape
            masked_hidden_2d = masked_hidden.reshape(B * M, D)  # [B*M, D]
            reconstruct_2d = self.mse_head(masked_hidden_2d)         # [B*M, D_out]，假设D_out = D
            reconstruct = reconstruct_2d.reshape(B, M, -1)       # [B, M, -1]
            original_patch = original_patch[:, mask_idx, :]
            reconstruct_loss = F.mse_loss(reconstruct, original_patch, reduction='mean')
        
        loss = None
        if label0 is not None and label1 is not None:
            ce_loss0 = criterion(logits0, label0)
            ce_loss1 = criterion(logits1, label1)
            loss = ce_loss1 + ce_loss0
        else:
            loss = 0.0
        #return logits0, logits1, hidden_states[:, 2:, :]
        return loss, reconstruct_loss, logits0, logits1

def train(model, 
        train_loader,
        device,
        global_step,
        logger,
        optimizer,
        warm_start=False,
        patch_mask=False,
        masked_rate=None,
        l1_lambda=0.0,
        rec_percent=1.1
        ):
    
    model.train() # Turn on the train mode  

    total_loss = 0.
    rec_loss = 0.
    epoch_train_acc_ = 0.
    class_weights = torch.tensor([1.0, 1.0, 2.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    f1_s = []
    for batch in train_loader:
        global_step += 1
        patch_tensors = batch["patch_tensors"].to(device)
        gene_tensors = batch["genes"].to(device)
        label0 = batch["label0"].to(device)
        label1 = batch["label1"].to(device)
        
        if patch_mask and masked_rate is not None:
            mask_tensors, mask_idx = token_mask(patch_tensors, masked_rate)
            mask_tensors.to(device)
            mask_idx.to(device)
            loss, reconstruct_loss, logits0, logits1 = model(
                mask_tensors, gene_tensors, label0, label1, patch_tensors, mask_idx
                # mask_ratio=mask_ratio
            )
        else:
            loss, reconstruct_loss, logits0, logits1 = model(
                mask_tensors, gene_tensors, label0, label1, criterion=criterion,
                # mask_ratio=mask_ratio
            )
        
        # Add MSE
        # rec_percent.to(device)
        loss = loss + rec_percent * reconstruct_loss
        # ---- L1  ----
        # l1_norm = 0.
        # if l1_lambda > 0:
        #     for p in model.parameters():
        #         l1_norm += p.abs().sum()
        
        # loss = loss + l1_lambda * l1_norm
    
        # compute accuracy for label1
        preds0 = torch.argmax(logits0, dim=-1)
        # acc1 = (preds0 == label0).float().mean().item()
        # epoch_train_acc_ += acc1
        
        stat_results = evaluate_multiclass_metrics(preds0, label0, 3)
        epoch_train_acc_+= stat_results['overall_accuracy']
        f1_s.append(stat_results["f1_scores"])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()
        rec_loss += reconstruct_loss.item()
    
    # dict collection
    total = sum((Counter(d) for d in f1_s), Counter())

    avg_f1_dict = {k: total[k] / len(f1_s) for k in total}
    return total_loss, rec_loss, global_step, epoch_train_acc_/len(train_loader), avg_f1_dict


def evaluate(model, 
        test_loader,
        device,
        global_step,
        logger,
        ):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    epoch_train_acc_ = 0.
    f1_s = []
    cm_list = []
    with torch.no_grad():
        # evaluating
        for batch in test_loader:
            patch_tensors = batch["patch_tensors"].to(device)
            gene_tensors = batch["genes"].to(device)
            label0 = batch["label0"].to(device)
            label1 = batch["label1"].to(device)
            loss, reconstruct_loss, logits0, logits1 = model(
                    patch_tensors, gene_tensors, label0, label1
                    # mask_ratio=mask_ratio
                )
            # compute accuracy for label1
            preds0 = torch.argmax(logits0, dim=-1)
            stat_results = evaluate_multiclass_metrics(preds0, label0, 3)
            epoch_train_acc_+= stat_results['overall_accuracy']
            f1_s.append(stat_results["f1_scores"])
            
            cm_list.append(stat_results['confusion_matrix'])
            logger.log(global_step, stat_results)

    total = sum((Counter(d) for d in f1_s), Counter())
    avg_f1_dict = {k: total[k] / len(f1_s) for k in total}
    acc_test = epoch_train_acc_/len(test_loader)
            
    return acc_test, avg_f1_dict


