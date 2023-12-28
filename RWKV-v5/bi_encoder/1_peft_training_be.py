import sys
import os
src_dir = os.path.dirname(os.path.dirname(__file__))
print(src_dir)
sys.path.append(src_dir)
from argparse import Namespace
from typing import Any
import torch
from peft import inject_adapter_in_model,LoraConfig,TaskType,PrefixTuningConfig,get_peft_model
import traceback
from datetime import datetime
import pytorch_lightning as pl
import math
import time
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import json
from functools import partial
def save_trainable_parameters(model, trainable_dir_output, model_filename,peft_config):
    print(f"save trainable parameters to {trainable_dir_output} pretrained from {model_filename}")
    # 创建保存目录
    os.makedirs(trainable_dir_output, exist_ok=True)
    
    # 获取可训练的参数
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    # 判断是否有可训练的参数
    if len(trainable_params) == 0:
        print("没有可训练的参数")
        return

    # 保存可训练的参数
    save_filename = os.path.basename(model_filename) + '_lora.pth'
    save_path = os.path.join(trainable_dir_output, save_filename)
    state_dict = {name: param.data for name, param in model.named_parameters() if param.requires_grad}
    torch.save(state_dict, save_path)
    print(f"save trainable parameters to {save_path}")

    config_filename = os.path.basename(model_filename) + '_peft.json'
    json_data = {}
    for key, value in peft_config.__dict__.items():
        try:
            if key == 'target_modules':
                value = list(value)
            json.dumps(value)
            json_data[key] = value
        except TypeError:
            pass

    # 将JSON对象导出到磁盘
    json_filename = os.path.basename(model_filename) + '_peft.json'
    json_filepath = os.path.join(trainable_dir_output, json_filename)
    with open(json_filepath, 'w') as json_file:
        json.dump(json_data, json_file)
    print(f"JSON对象已导出到磁盘：{json_filepath}")

def load_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            import gc
            gc.collect()
            n_embd = w['emb.weight'].shape[1]
            vocab_size = w['emb.weight'].shape[0]
            dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            n_layer = 0
            keys = list(w.keys())
            version = 4
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                n_layer = max(n_layer, layer_id+1)
                if 'ln_x' in x:
                    version = max(5, version)
                if 'gate.weight' in x:
                    version = max(5.1, version)
                if int(version) == 5 and 'att.time_decay' in x:
                    n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            version = max(5.2, version)
            head_size_a = dim_att // n_head
            args.n_embd = n_embd
            args.dim_att = dim_att
            args.dim_ffn = dim_ffn
            args.n_layer = n_layer
            args.version = version
            args.head_size_a = head_size_a
            args.vocab_size = vocab_size
            return w
    except Exception as e:
        traceback.print_exc()
        return None

class YueyuTrainCallback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay



        # rank_zero_info(f"{real_step} {lr}")

        if trainer.is_global_zero:
            if  trainer.global_step == 0: # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging   
            if batch_idx % args.save_per_batches == 0:
                print(f'saving trainable to {args.trainable_dir_output}')
                output_dir = f"{args.trainable_dir_output}_{batch_idx}"
                save_trainable_parameters(pl_module, output_dir, args.model_file,args.peft_config)
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss,   "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f}  {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="/Users/yueyulin/Downloads/RWKV-5-World-1B5-v2-20231025-ctx4096.pth", help='model file')
    parser.add_argument('--ds_dir', type=str, default="/media/yueyulin/bigdata/ds/mmarco_chinese/rwkv_tokenized_ids", help='epoch begin')
    parser.add_argument('--ctx_len', type=int, default=512)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--tuner', type=str, default='lora', help='tuner type', choices=['lora'])
    parser.add_argument('--output_dir', type=str, default=None, help='peft output dir')
    parser.add_argument('--device', type=str, default="cuda", help='trainer device')
    parser.add_argument('--lora_ckpt', type=str, default=None, help='lora ckpt')
    cmd_args = parser.parse_args()
    model_file = cmd_args.model_file
    args = Namespace()
    w = load_ckpt_and_parse_args(model_file, args)
    device = cmd_args.device
    print(w.keys())
    print(args)
    args.model_file = model_file
    args.my_pos_emb = 0
    args.pre_ffn = 0
    args.head_size_divisor = 8
    args.ctx_len = cmd_args.ctx_len
    args.dropout = 0
    args.head_qk = 0
    args.grad_cp = 0
    args.save_per_batches = 10000
    args.my_exit = 3
    import os
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    if torch.backends.mps.is_available():
        os.environ['RWKV_FLOAT_MODE'] = 'fp32'
        precision = "fp32"
    elif torch.cuda.is_available():
        os.environ['RWKV_FLOAT_MODE'] = 'bf16'
        precision = "bf16"
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    from src.model import RWKV
    from src.model_for_sequence_embedding import RwkvForSequenceEmbedding

    model = RWKV(args)
    model.load_state_dict(w)
    del w
    if cmd_args.tuner =='lora':
        max_steps = 0
        lora_weights = None
        if cmd_args.lora_ckpt is None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                target_modules=["ffn.receptance","ffn.key","ffn.value","att.receptance","att.key","att.value"],)
            model = inject_adapter_in_model(lora_config, model)
            print(model)
            args.peft_config = lora_config
            # model = get_peft_model(model, peft_config)
            # print(model)
            # exit(0)
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            args.from_steps = 0
        else:
            dirs = os.listdir(cmd_args.lora_ckpt)
            max_steps = 0
            for dir in dirs:
                #dir looks like trainable_model_70000
                if os.path.isdir(os.path.join(cmd_args.lora_ckpt,dir)):
                    if dir.startswith('trainable_model_'):
                        steps = int(dir.split('_')[-1])
                        if steps > max_steps:
                            max_steps = steps
            print('max_steps',max_steps)
            base_model_file = os.path.basename(model_file)
            lora_ckpt_file = os.path.join(cmd_args.lora_ckpt,f'trainable_model_{max_steps}/{base_model_file}_lora.pth')
            lora_config_file = os.path.join(cmd_args.lora_ckpt,f'trainable_model_{max_steps}/{base_model_file}_peft.json')
            print('load lora ckpt from',lora_ckpt_file, ' with config ',lora_config_file)
            with open(lora_config_file) as f:
                lora_config = json.load(f)
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    lora_alpha=lora_config['lora_alpha'],
                    lora_dropout=lora_config['lora_dropout'],
                    r=lora_config['r'],
                    bias=lora_config['bias'],
                    target_modules=lora_config['target_modules'],)
            model = inject_adapter_in_model(lora_config, model)
            lora_weights = torch.load(lora_ckpt_file)
            args.peft_config = lora_config
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            args.from_steps = max_steps+1
    print('---------------------------------------')
    model = RwkvForSequenceEmbedding(model)
    print(model)
    if lora_weights is not None:
        inform = model.load_state_dict(lora_weights,strict=False)
        print("\033[92m", inform, "\033[0m")
        del lora_weights

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("可训练参数数量：", trainable_params)
    print("总参数数量：", total_params)
    print("比例：", trainable_params / total_params)
    model = model.bfloat16()
    model = model.to(device)
    data_file = cmd_args.ds_dir
    from datasets_utilities import load_cross_encoder_ds_from_disk
    train_data = load_cross_encoder_ds_from_disk(data_file,cmd_args.ctx_len)
    print(train_data['train'][0])
    print(train_data['train'][1])
    def collate_fn(examples):
        query = torch.tensor([x['query'] for x in examples], dtype=torch.long)
        positive = torch.tensor([x['positive'] for x in examples], dtype=torch.long)
        negative = torch.tensor([x['negative'] for x in examples], dtype=torch.long)
        logits_positive = torch.tensor([x['logits_positive'] for x in examples], dtype=torch.float)
        logits_negative = torch.tensor([x['logits_negative'] for x in examples], dtype=torch.float)
        return {
            "query": query,
            "positive": positive,
            "negative": negative,
            "logits_positive": logits_positive,
            "logits_negative": logits_negative
        }
    data_loader = DataLoader(train_data['train'], shuffle=True, pin_memory=True, batch_size=cmd_args.bs, num_workers=24, persistent_workers=True, drop_last=True, collate_fn=collate_fn)
    # d = next(iter(data_loader))
    # print(d)
    # print(d[0].shape)

    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.num_sanity_val_steps = 0
    args.enable_checkpointing = False
    args.accumulate_grad_batches = 1
    args.gradient_clip_val = 1.0
    args.lr_final = 1e-5
    args.lr_init = 3e-4
    args.warmup_steps = 50
    args.beta1 = 0.9
    args.beta2 = 0.99
    args.betas = (args.beta1, args.beta2)
    args.adam_eps = 1e-8
    args.weight_decay = 0.01
    args.weight_decay_final = -1
    args.precision = precision
    args.logger = False
    args.my_pile_stage = 0
    args.my_pile_edecay = 0
    args.layerwise_lr = 1
    args.epoch_begin = 0
    args.epoch_count = 100 
    args.epoch_save = 1
    args.epoch_steps = 1000
    args.max_epochs = args.epoch_count
    args.my_exit_tokens = 0
    args.proj_dir = cmd_args.output_dir
    args.my_timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.wandb = 'rwkv5_bi_encoder_att_ffn'
    args.run_name = 'yy' 
    args.my_qa_mask = 0
    args.num_nodes = 1
    args.devices = 1
    args.micro_bsz = cmd_args.bs 
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    args.magic_prime = 0
    args.trainable_dir_output = os.path.join(args.proj_dir, "trainable_model")
    os.makedirs(args.proj_dir, exist_ok=True)
    os.makedirs(args.trainable_dir_output, exist_ok=True)


   


    

    trainer = Trainer(accelerator=device,strategy="auto",devices=1,num_nodes=1,precision=precision,
            logger=args.logger,callbacks=[YueyuTrainCallback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
            log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)

    

    trainer.fit(model, data_loader)
