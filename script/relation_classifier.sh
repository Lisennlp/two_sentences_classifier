#!/bin/bash

if [ "$1" = "train-big" ]; then
        echo "start to train big data......"

        CUDA_VISIBLE_DEVICES=4,5,6 python ../two_sentences_classifier/add_type_train.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/xd/data/novels/figure_relation/train/  \
                                  --eval_train_file  /nas/xd/data/novels/figure_relation/train/figure_relation_6001  \
                                  --eval_file  /nas/xd/data/novels/figure_relation/dev/figure_relation_5001   \
                                  --train_batch_size 25   \
                                  --eval_batch_size 5 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./drop_0.1_activate_cls_abs_model0530 \
                                  --reduce_dim 768  \
                                  --gpu0_size 1  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 5 3>&2 2>&1 1>&3 | tee logs/drop_0.1_activate_cls_abs_model0530.log


elif [ "$1" = "train-small" ];then
        echo "start to train small data......"

        CUDA_VISIBLE_DEVICES=1,2,3 python ../two_sentences_classifier/add_type_train.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/xd/data/novels/figure_relation/small/small.data.train  \
                                  --eval_train_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_6001  \
                                  --eval_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_5001   \
                                  --train_batch_size  25 \
                                  --eval_batch_size 5 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 7   \
                                  --num_labels 2   \
                                  --output_dir ./drop_0.3_activate_cls_abs_model0529 \
                                  --reduce_dim 768  \
                                  --gpu0_size 1  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 5 3>&2 2>&1 1>&3 | tee logs/drop_0.3_activate_cls_abs_model0529.log

elif [ "$1" = "predict-small" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=5,6,7 python ../two_sentences_classifier/add_type_train.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/xd/data/novels/figure_relation/small/small.data.train \
                                        --eval_train_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_6001  \
                                        --eval_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_5001   \
                                        --train_batch_size 30   \
                                        --eval_batch_size 5 \
                                        --learning_rate 3e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 7   \
                                        --num_labels 2   \
                                        --result_file ./activate_cls_abs_model0525/predict0528_dev.csv  \
                                        --output_dir ./activate_cls_abs_model0525  \
                                        --reduce_dim 768  \
                                        --gpu0_size 1  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

elif [ "$1" = "predict-big" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=5,6,7 python ../two_sentences_classifier/add_type_train.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/xd/data/novels/figure_relation/small/small.data.train \
                                        --eval_train_file  /nas/xd/data/novels/figure_relation/small/small.figure_relation_6001  \
                                        --eval_file  /nas/xd/data/novels/figure_relation/dev/figure_relation_5001   \
                                        --train_batch_size 30   \
                                        --eval_batch_size 5 \
                                        --learning_rate 3e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 7   \
                                        --num_labels 2   \
                                        --result_file ./activate_cls_abs_model0525/predict0528_dev.csv  \
                                        --output_dir ./activate_cls_abs_model0525  \
                                        --reduce_dim 768  \
                                        --gpu0_size 1  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

else
    echo 'unknown argment 1'
fi
