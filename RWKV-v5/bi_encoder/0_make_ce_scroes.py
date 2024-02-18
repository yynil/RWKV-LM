import sys
import os
src_dir = os.path.dirname(os.path.dirname(__file__))
print(src_dir)
sys.path.append(src_dir)

from src.model_for_classification import RwkvForClassification_Run
import torch
import traceback
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

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
            args.n_head = args.dim_att // args.head_size_a

            #####Fixed args
            args.head_qk = 0
            args.dropout = 0
            args.my_pos_emb = 0
            args.pre_ffn = 0
            args.head_size_divisor = 8
            
            return w
    except Exception as e:
        traceback.print_exc()
        return None

def load_full_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            import gc
            gc.collect()
            
            n_embd = w['rwkvModel.emb.weight'].shape[1]
            vocab_size = w['rwkvModel.emb.weight'].shape[0]
            dim_att = w['rwkvModel.blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['rwkvModel.blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            num_labels = w['score.weight'].shape[0]
            n_layer = 0
            keys = list(w.keys())
            version = 4
            for x in keys:
                layer_id = int(x.split('.')[2]) if ('blocks.' in x) else 0
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
            args.n_head = args.dim_att // args.head_size_a
            args.num_labels = num_labels
            #####Fixed args
            args.head_qk = 0
            args.dropout = 0
            args.my_pos_emb = 0
            args.pre_ffn = 0
            args.head_size_divisor = 8
            return w
    except Exception as e:
        traceback.print_exc()
        return None
    

def inference(model: RwkvForClassification_Run, template: str, tokenizer :TRIE_TOKENIZER,query :str, document :str):
    cls_id = 1
    sep_id = 2
    # input_str = template.format(query=query,document=document)
    # input_ids = tokenizer.encode(input_str)+[cls_id]
    input_ids = tokenizer.encode(query)+[sep_id]+tokenizer.encode(document)+[cls_id]
    from torch.amp import autocast
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        logits = model(torch.tensor([input_ids]).long())
    return logits

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='lora', choices=['full', 'lora'])
    parser.add_argument('--ckpt', type=str, default='/home/gpu/data/sdb1/models/rwkv/RWKV-5-World-7B-v2-20240128-ctx4096.pth')
    parser.add_argument('--lora_ckpt', type=str, default='/home/gpu/data/sdb1/models/rwkv/lora/RWKV5_7B/ce_att_ffn_1k_mmarcro_chinese/trainable_model_90000')
    parser.add_argument('--triple_file',type=str,default="/home/gpu/data/sdb1/downloads/google_data/triples.train.ids.small.tsv")
    parser.add_argument('--output_dir',type=str,default="/home/gpu/data/sdb1/downloads/google_data")
    parser.add_argument('--start_index',type=int,default=0)
    parser.add_argument('--end_index',type=int,default=1000000)
    parser.add_argument('--device',type=str,default='cuda:0')
    template = None 
    args = parser.parse_args()
    if args.type == 'full':
        w  = load_full_ckpt_and_parse_args(args.ckpt, args)
        from src.model_run import RWKV
        model = RWKV(args)
        model = RwkvForClassification_Run(model, args.num_labels)
        inform = model.load_state_dict(w,strict=False)
        print(model)
        print(inform)
        query = "中华人民共和国是什么时候成立的？"
        document_positive = "中华人民共和国（the People's Republic of China），简称“中国”，成立于1949年10月1日 [1]，位于亚洲东部，太平洋西岸 [2]，是工人阶级领导的、以工农联盟为基础的人民民主专政的社会主义国家 [3]，以五星红旗为国旗 [4]、《义勇军进行曲》为国歌 [5]，国徽中间是五星照耀下的天安门，周围是谷穗和齿轮 [6] [170]，通用语言文字是普通话和规范汉字 [7]，首都北京 [8]，是一个以汉族为主体、56个民族共同组成的统一的多民族国家。中国陆地面积约960万平方千米，东部和南部大陆海岸线1.8万多千米，海域总面积约473万平方千米 [2]。海域分布有大小岛屿7600多个，其中台湾岛最大，面积35798平方千米 [2]。中国同14国接壤，与8国海上相邻。省级行政区划为23个省、5个自治区、4个直辖市、2个特别行政区。 [2]"
        document_negative = "河北省，简称“冀”，是中华人民共和国省级行政区，省会石家庄，位于北纬36°05′-42°40′，东经113°27′-119°50′之间，环抱首都北京市，东与天津市毗连并紧傍渤海，东南部、南部衔山东省、河南省，西倚太行山与山西为邻，西北部、北部与内蒙古自治区交界，东北部与辽宁省接壤，总面积18.88万平方千米。 [1-2]河北省下辖11个地级市，共有49个市辖区、21个县级市、91个县、6个自治县。 [4-5]截至2022年末，河北省常住人口为7420万人。 [3] [132]"

        
        import os
        tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
        model = model.bfloat16()
        model = model.to('cuda')
        model.eval()

        inference(model, template, tokenizer, query, document_positive)
        inference(model, template, tokenizer, query, document_negative)
    elif args.type == 'lora':
        w = load_ckpt_and_parse_args(args.ckpt, args)
        from src.model_run import RWKV
        model = RWKV(args)
        inform = model.load_state_dict(w,strict=False)
        del w 
        #list lora_ckpt as directory
        import os
        if not os.path.isdir(args.lora_ckpt):
            print('lora_ckpt should be a directory')
            exit(-1)
        files = os.listdir(args.lora_ckpt)
        ckpt_file = None
        lora_config = None
        for file in files:
            if file.endswith('.pth'):
                ckpt_file = os.path.join(args.lora_ckpt,file)
            elif file.endswith('.json'):
                lora_config = os.path.join(args.lora_ckpt,file)
            
            if ckpt_file is not None and lora_config is not None:
                break
        print('load ckpt from ',ckpt_file,' and config from ',lora_config)
        w = torch.load(ckpt_file, map_location='cpu')
        num_labels = w['score.weight'].shape[0]
        import json
        from peft import LoraConfig,TaskType,inject_adapter_in_model
        with open(lora_config,'r') as f:
            lora_obj = json.load(f)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=lora_obj['lora_alpha'],
                lora_dropout=0,
                r=lora_obj['r'],
                bias=lora_obj['bias'],
                target_modules=lora_obj['target_modules'],)
            print(lora_config)
        model = inject_adapter_in_model(lora_config,model)
        model = RwkvForClassification_Run(model, num_labels,device=args.device)
        print(model)
        inform = model.load_state_dict(w,strict=False)
        import os
        tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
        model = model.bfloat16()
        model = model.to(args.device)
        model.eval()
        query = "中华人民共和国是什么时候成立的？"
        document_positive = "中华人民共和国（the People's Republic of China），简称“中国”，成立于1949年10月1日 [1]，位于亚洲东部，太平洋西岸 [2]，是工人阶级领导的、以工农联盟为基础的人民民主专政的社会主义国家 [3]，以五星红旗为国旗 [4]、《义勇军进行曲》为国歌 [5]，国徽中间是五星照耀下的天安门，周围是谷穗和齿轮 [6] [170]，通用语言文字是普通话和规范汉字 [7]，首都北京 [8]，是一个以汉族为主体、56个民族共同组成的统一的多民族国家。中国陆地面积约960万平方千米，东部和南部大陆海岸线1.8万多千米，海域总面积约473万平方千米 [2]。海域分布有大小岛屿7600多个，其中台湾岛最大，面积35798平方千米 [2]。中国同14国接壤，与8国海上相邻。省级行政区划为23个省、5个自治区、4个直辖市、2个特别行政区。 [2]"
        document_negative = "河北省，简称“冀”，是中华人民共和国省级行政区，省会石家庄，位于北纬36°05′-42°40′，东经113°27′-119°50′之间，环抱首都北京市，东与天津市毗连并紧傍渤海，东南部、南部衔山东省、河南省，西倚太行山与山西为邻，西北部、北部与内蒙古自治区交界，东北部与辽宁省接壤，总面积18.88万平方千米。 [1-2]河北省下辖11个地级市，共有49个市辖区、21个县级市、91个县、6个自治县。 [4-5]截至2022年末，河北省常住人口为7420万人。 [3] [132]"
        print(inference(model, template, tokenizer, query, document_positive))
        print(inference(model, template, tokenizer, query, document_negative))

    

    from tqdm import tqdm
    from datasets import load_dataset
    if args.triple_file.endswith('.tsv'):
        cross_encoder_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cross_encoder')
        script_file = os.path.join(cross_encoder_dir, "cross_encoder_ds.py")
        cache_dir = os.path.join(cross_encoder_dir, "cache")
        dataset = load_dataset(script_file,"chinese", data_dir="/home/gpu/data/sdb1/downloads/google_data",cache_dir=cache_dir)
    elif args.triple_file.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=args.triple_file)
    print(dataset['train'][1])

    length = len(dataset['train'])
    print(length)
    args.end_index = min(args.end_index+args.start_index,length)
    progress = tqdm(range(args.start_index,args.end_index), desc='Inference')
    os.makedirs(args.output_dir,exist_ok=True)
    output_file = os.path.join(args.output_dir,f'ce_scores_{args.start_index}_2_{args.end_index}.jsonl')
    import json
    with open(output_file,'w',buffering=1024*1024*20,encoding='UTF-8') as f:
        for i in progress:
            example = dataset['train'][i]
            query = example['query']
            positive = example['positive']
            negative = example['negative']
            logits_positive = inference(model, template, tokenizer, query, positive).item()
            logits_negative = inference(model, template, tokenizer, query, negative).item()
            progress.set_description(f'Inference {i}')
            json_str = json.dumps({"query":query,"positive":positive,"negative":negative,"logits_positive":logits_positive,"logits_negative":logits_negative},ensure_ascii=False)
            f.write(json_str+'\n')