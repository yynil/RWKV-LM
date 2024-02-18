# 推荐训练配置
3090ti 24G GRAM
训练数据mmacro
策略：deepspeed_stage_2_offload
base model rwkv 7b
ctx_len 256
bs 4

python cross_encoder/1_peft_training_ce.py --model_file /media/yueyulin/bigdata/models/rwkv5/rwkv-x052-7b-world-v2-79%trained-20231208-ctx4k.pth --ctx_len 256 --bs 4 --ds_dir /home/yueyulin/ds/rwkv_tokenized_ids/ --output_dir /media/yueyulin/bigdata/models/lora/rwkv7b/ce_att_ffn/ --lora_ckpt /media/yueyulin/bigdata/models/lora/rwkv7b/ce_att_ffn/