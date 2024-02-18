import torch
import torch.nn as nn
from torch.nn import Module

import transformers
import deepspeed

import os
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
from sentence_transformers.util import pairwise_dot_score
class RwkvForSequenceEmbedding(pl.LightningModule):

    def __init__(self, rwkvModel,pad_id = 0):
        super(RwkvForSequenceEmbedding, self).__init__()
        self.pad_id = pad_id
        self.rwkvModel = rwkvModel
        del self.rwkvModel.head
    def forward(self, idx):
        args = self.rwkvModel.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.rwkvModel.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.rwkvModel.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.rwkvModel.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.rwkvModel.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.rwkvModel.ln_out(x)

        #calculate the idx actual length which is first self.pad_id
        idx_actual_len = torch.eq(idx, 1).int().argmax(-1)
        x = x[torch.arange(B), idx_actual_len]
        return x
    
    def configure_optimizers(self) :
        args = self.rwkvModel.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                if ("time_mix" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_decay" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif ("time_first" in n) and (args.layerwise_lr > 0):
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        print('optim_groups', optim_groups)
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if torch.backends.mps.is_available():
                from torch.optim import AdamW,Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                return DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True,  weight_decay=args.weight_decay, amsgrad=False)
                # return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if torch.backends.mps.is_available():
                from torch.optim import AdamW, Adam,SGD
                # return SGD(optim_groups, lr=self.args.lr_init, momentum=0.9, weight_decay=0)
                return Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,  weight_decay=0, amsgrad=False)
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False)
                # return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, adam_w_mode=False, weight_decay=0, amsgrad=False)
    
    def training_step(self, batch, batch_idx):
        query = batch["query"]
        positive = batch["positive"]
        negative = batch["negative"]
        logits_positive = batch["logits_positive"]
        logits_negative = batch["logits_negative"]
        query_embeddings = self(query)
        positive_embeddings = self(positive)
        negative_embeddings = self(negative)
        labels = logits_positive - logits_negative
        positive_scores = pairwise_dot_score(query_embeddings, positive_embeddings)
        negative_scores = pairwise_dot_score(query_embeddings, negative_embeddings)
        loss_fct = nn.MSELoss()
        return loss_fct(positive_scores-negative_scores, labels)
    
class RwkvForSequenceEmbedding_Run(pl.LightningModule):
    def __init__(self, rwkvModel, device = 'cuda',chunk_size=128,delete_head=False):
        super(RwkvForSequenceEmbedding_Run, self).__init__()
        self.rwkvModel = rwkvModel
        if hasattr(self.rwkvModel, 'head') and delete_head:
            del self.rwkvModel.head
        else:
            print("self.rwkvModel does not have a 'head' attribute or delete_head is False [{delete_head}]")
        self.my_device = device
        self.chunk_size = chunk_size

    def eval(self) :
        super().eval()
        self.rwkvModel.emb = self.rwkvModel.emb.to('cpu')

    def forward(self, idx,state=None):

        with torch.no_grad():
            if isinstance(idx,torch.Tensor):
                idx_shape = idx.shape
                if len(idx_shape) == 2:
                    idx = idx.squeeze(0)
                idx = idx.to('cpu')
            elif isinstance(idx,list):
                idx = torch.tensor(idx,device='cpu',requires_grad=False)
            if idx[-1].item() != 1:
                idx = torch.cat([idx,torch.tensor([1],device='cpu',requires_grad=False)])
            if state is None:
                state = [None] * self.rwkvModel.n_layer * 3
                for i in range(self.rwkvModel.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(self.rwkvModel.n_embd, dtype=torch.float32, requires_grad=False, device=self.my_device).contiguous()
                    state[i*3+1] = torch.zeros((self.rwkvModel.n_head, self.rwkvModel.head_size_a, self.rwkvModel.head_size_a), dtype=torch.float32, requires_grad=False, device=self.my_device).contiguous()
                    state[i*3+2] = torch.zeros(self.rwkvModel.n_embd, dtype=torch.float32, requires_grad=False, device=self.my_device).contiguous()
            offset = 0
            while offset < idx.shape[0]:
                idx_chunk = idx[offset:offset+self.chunk_size]
                x = self.rwkvModel.emb(idx_chunk).to(self.my_device)

                for i, block in enumerate(self.rwkvModel.blocks):
                    x,x_x,state_kv,state_ffn = block(x,state[i*3:i*3+3])
                    state[i*3+0] = x_x
                    state[i*3+1] = state_kv
                    state[i*3+2] = state_ffn

                x = self.rwkvModel.ln_out(x)
                offset += self.chunk_size
            # x = self.rwkvModel.emb(idx).to(self.my_device)

            # for i, block in enumerate(self.rwkvModel.blocks):
            #     x,x_x,state_kv,state_ffn = block(x,state[i*3:i*3+3])
            #     state[i*3+0] = x_x
            #     state[i*3+1] = state_kv
            #     state[i*3+2] = state_ffn

            # x = self.rwkvModel.ln_out(x)
            #calculate the idx actual length which is first self.pad_id
            # idx_actual_len = torch.eq(idx, 1).int().argmax(-1)
            x = x[-1]
            return x