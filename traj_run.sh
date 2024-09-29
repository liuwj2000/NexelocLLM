#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --gpu_ids 'all' \
                            --mixed_precision bf16 \
                            --num_processes 8 run_model_NextlocLLM_MER.py \
                            --config_file user_config/xian\
                            --max_epoch 100\
                            --llm_model "gpt2" \
                            --learning_rate 0.0002 \
                            --batch_size 128\
                            --if_prompt 1 \
                            --if_sim 0  \
                            --if_dur_loss 0 \
                            --if_dur_emb 1 \
                            --if_lora 0\
                            --dropout_p 0.5 \
                            --if_llm_poi 1 \
                            --if_train 1 \
                            --save_dir acce_file/Xian_3 >Xian_7.log 2>&1