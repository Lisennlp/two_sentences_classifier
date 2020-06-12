#!/bin/bash

if [ "$1" = "train-big" ]; then
        echo "start to train big data......"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../two_sentences_classifier/relation_classify.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/lishengping/datas/figure_relation_lsp_final/train/  \
                                  --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/train/figure_relation_2281  \
                                  --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/dev/data.dev   \
                                  --train_batch_size 30   \
                                  --eval_batch_size 15 \
                                  --learning_rate 3e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./768_cls_mean_0612 \
                                  --reduce_dim 768  \
                                  --gpu0_size 1  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 2 3>&2 2>&1 1>&3 | tee logs/768_cls_mean_0612.log


elif [ "$1" = "train-small" ];then
        echo "start to train small data......"

        CUDA_VISIBLE_DEVICES=0,1,2,3 python ../two_sentences_classifier/relation_classify.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/lishengping/datas/figure_relation_lsp_final/small/small.data.train  \
                                  --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                  --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_1   \
                                  --train_batch_size  28 \
                                  --eval_batch_size 7 \
                                  --learning_rate 3e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./768_cls_mean_0605 \
                                  --reduce_dim 768  \
                                  --gpu0_size 1  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 4 3>&2 2>&1 1>&3 | tee logs/768_cls_mean_0605.log

elif [ "$1" = "predict-small" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../two_sentences_classifier/add_type_train.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/lishengping/datas/figure_relation_lsp_final/small/small.data.train \
                                        --eval_train_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_6001  \
                                        --eval_file  /nas/lishengping/datas/figure_relation_lsp_final/small/small.figure_relation_5001   \
                                        --train_batch_size 30   \
                                        --eval_batch_size 80 \
                                        --learning_rate 3e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 20   \
                                        --num_labels 2   \
                                        --result_file ./predict0602_15_20_dev.csv  \
                                        --output_dir /nas/lishengping/relation_models/activate_cls_abs_model0531_15  \
                                        --reduce_dim 768  \
                                        --gpu0_size 0  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

elif [ "$1" = "predict-big" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=3,4 python ../two_sentences_classifier/add_type_train.py \
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
