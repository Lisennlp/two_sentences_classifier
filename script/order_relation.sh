#!/bin/bash

if [ "$1" = "train-big" ]; then
        echo "start to train big data......"

        CUDA_VISIBLE_DEVICES=0,1 python ../relation_classifier/order_relation_classifier2.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/lishengping/datas/figure_relation_lsp_final/train/  \
                                  --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/train/figure_relation_2281  \
                                  --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/dev/data.dev   \
                                  --train_batch_size 21   \
                                  --eval_batch_size 3 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 4   \
                                  --output_dir ./order_b21_lr1e5_t7_d03_relation_0619 \
                                  --reduce_dim 768  \
                                  --gpu0_size 1  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 7 3>&2 2>&1 1>&3 | tee logs/order_b21_lr1e5_t7_d03_relation_0619.log


elif [ "$1" = "train-small" ];then
        echo "start to train small data......"

        CUDA_VISIBLE_DEVICES=0,1 python ../relation_classifier/order_relation_classifier2.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/lishengping/datas/figure_relation_lsp_final/small/small.data.train  \
                                  --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                  --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_1   \
                                  --train_batch_size 21 \
                                  --eval_batch_size 3 \
                                  --learning_rate 3e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./small_relation_0616 \
                                  --reduce_dim 768  \
                                  --gpu0_size 1  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 7 3>&2 2>&1 1>&3 | tee logs/small_relation_0616.log

elif [ "$1" = "eval-small" ];then
        echo "start to eval......"

        CUDA_VISIBLE_DEVICES=0,1 python ../relation_classifier/order_relation_classifier.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_2281 \
                                        --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                        --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_1   \
                                        --train_batch_size  24  \
                                        --eval_batch_size 4 \
                                        --learning_rate 1e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 7   \
                                        --num_labels 4   \
                                        --result_file ./order_relation.csv  \
                                        --output_dir order_b27_lr1e5_t7_d03_relation_0615  \
                                        --reduce_dim 768  \
                                        --gpu0_size 0  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_eval   \
                                        --gradient_accumulation_steps 12 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

elif [ "$1" = "predict-small" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=0,1 python ../relation_classifier/order_relation_classifier.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_2281 \
                                        --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                        --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_1   \
                                        --train_batch_size  24  \
                                        --eval_batch_size 4 \
                                        --learning_rate 1e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 7   \
                                        --num_labels 4   \
                                        --result_file ./order_relation.csv  \
                                        --output_dir order_b27_lr1e5_t7_d03_relation_0615  \
                                        --reduce_dim 768  \
                                        --gpu0_size 0  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 12 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

elif [ "$1" = "predict-big" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=3,4 python ../relation_classifier/order_relation_classifier.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/lishengping/datas/figure_relation_lsp_final/small/small.data.train \
                                        --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_6001  \
                                        --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/dev/figure_relation_5001   \
                                        --train_batch_size 30   \
                                        --eval_batch_size 20 \
                                        --learning_rate 3e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 15   \
                                        --num_labels 2   \
                                        --result_file ./predict0602_15_15_dev.csv  \
                                        --output_dir /nas/lishengping/relation_models/activate_cls_abs_model0531_15  \
                                        --reduce_dim 768  \
                                        --gpu0_size 0  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

else
    echo 'unknown argment 1'
fi
