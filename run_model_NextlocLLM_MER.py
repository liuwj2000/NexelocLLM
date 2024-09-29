"""
训练并评估单一模型的脚本
"""

import argparse

from libcity.pipeline import run_model_NextlocLLM_MER_lora,test_model_NextlocLLM_MER_lora
from libcity.utils import str2bool, add_general_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device',  type=str, default='cuda:2', help='one gpu device')
    parser.add_argument('--llm_model',  type=str, default='gpt2')
    parser.add_argument('--if_prompt',  type=int, default=1)
    parser.add_argument('--if_sim',  type=int, default=0)
    parser.add_argument('--if_dur_emb',  type=int, default=0)
    parser.add_argument('--if_dur_loss',  type=int, default=0)
    parser.add_argument('--if_lora',  type=int, default=0)
    parser.add_argument('--if_llm_poi',  type=int, default=0)
    parser.add_argument('--dim_feature',  type=int, default=128)
    parser.add_argument('--num_layer',  type=int, default=6)
    parser.add_argument('--dropout_p',  type=float, default=0.5)
    parser.add_argument('--mer_dim',  type=int, default=128)
    parser.add_argument('--day_dim',  type=int, default=16)
    parser.add_argument('--hour_dim',  type=int, default=16)
    parser.add_argument('--dur_dim',  type=int, default=16)
    parser.add_argument('--poi_dim',  type=int, default=16)
    parser.add_argument('--if_train',  type=int, default=1)
    parser.add_argument('--if_poi',  type=int, default=1)
    parser.add_argument('--save_dir',type=str)
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()

    if args.if_llm_poi == 1:
        args.if_llm_poi = True
    else:
        args.if_llm_poi = False

    if args.if_prompt == 1:
        args.if_prompt = True
    else:
        args.if_prompt = False

    if args.if_sim == 1:
        args.if_sim = True
    else:
        args.if_sim = False
    
    if args.if_dur_emb == 1:
        args.if_dur_emb = True
    else:
        args.if_dur_emb = False

    if args.if_dur_loss == 1:
        args.if_dur_loss = True
    else:
        args.if_dur_loss = False
    
    if args.if_lora == 1:
        args.if_lora = True
    else:
        args.if_lora = False
        
    if args.if_poi == 1:
        args.if_poi = True
    else:
        args.if_poi = False

    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    #print(args.if_prompt)
    if(args.if_train):
        run_model_NextlocLLM_MER_lora(
                        config_file=args.config_file, 
                        other_args=other_args)
    else:
        test_model_NextlocLLM_MER_lora(
                        config_file=args.config_file, 
                        other_args=other_args)
