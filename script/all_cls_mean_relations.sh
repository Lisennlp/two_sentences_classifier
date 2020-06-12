#!/bin/bash

if [ "$1" = "train-big" ]; then
        echo "start to train big data......"

        CUDA_VISIBLE_DEVICES=2,3 python ../two_sentences_classifier/all_cls_mean_relations.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/xd/data/novels/figure_relation/train/  \
                                  --eval_train_file  /nas/xd/data/novels/figure_relation/train/figure_relation_6001  \
                                  --eval_file  /nas/xd/data/novels/figure_relation/dev/figure_relation_5001   \
                                  --train_batch_size 20   \
                                  --eval_batch_size 10 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./all_token_mean_abs_model0528 \
                                  --reduce_dim 768  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 2 3>&2 2>&1 1>&3 | tee logs/all_token_mean_abs_model0528.log


elif [ "$1" = "train-small" ];then
        echo "start to train small data......"


        CUDA_VISIBLE_DEVICES=2,3 python ../two_sentences_classifier/all_cls_mean_relations.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/xd/data/novels/figure_relation/small/small.data.train  \
                                  --eval_train_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_6001  \
                                  --eval_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_5001   \
                                  --train_batch_size  20 \
                                  --eval_batch_size 10 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./all_token_mean_abs_model0528 \
                                  --reduce_dim 768  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 2 3>&2 2>&1 1>&3 | tee logs/all_token_mean_abs_model0528.log

elif [ "$1" = "predict" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=5,6,7 python ../two_sentences_classifier/all_cls_mean_relations.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/xd/data/novels/figure_relation/data.train \
                                        --eval_train_file  /nas/xd/data/novels/figure_relation/2001.train  \
                                        --eval_file  /nas/xd/data/novels/figure_relation/data.dev   \
                                        --train_batch_size 30   \
                                        --eval_batch_size 5 \
                                        --learning_rate 3e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 7   \
                                        --num_labels 2   \
                                        --result_file ./add_type_model0515_7_1/predict0518_dev.csv  \
                                        --output_dir ./add_type_model0515_7_1  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/add_type_model0515_7_1.log

else
    echo 'unknown argment 1'
fi
