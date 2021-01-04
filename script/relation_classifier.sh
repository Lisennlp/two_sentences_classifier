#!/bin/bash

if [ "$1" = "train-big" ]; then
        echo "start to train big data......"

        CUDA_VISIBLE_DEVICES=1,0 python ../relation_classifier/relation_classify.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/lishengping/datas/person2vector/figure_relation_lsp_final/train/  \
                                  --eval_train_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/train/figure_relation_2281  \
                                  --eval_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/dev/data.dev   \
                                  --train_batch_size 24   \
                                  --eval_batch_size 12 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 10   \
                                  --num_labels 2   \
                                  --output_dir ./cat_b24_lr1e5_t10_d03_relation_1223 \
                                  --reduce_dim 768  \
                                  --gpu0_size 0  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/cat_b24_lr1e5_t10_d03_relation_1223.log


elif [ "$1" = "train-small" ];then
        echo "start to train small data......"

        CUDA_VISIBLE_DEVICES=1,0 python ../relation_classifier/relation_classify.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.data.train  \
                                  --eval_train_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                  --eval_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.figure_relation_1   \
                                  --train_batch_size  24 \
                                  --eval_batch_size  6 \
                                  --learning_rate 1e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 10   \
                                  --num_labels 2  \
                                  --output_dir ./small_relation_0701 \
                                  --reduce_dim 768  \
                                  --gpu0_size 0  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/small_relation_1223.log

elif [ "$1" = "predict-small" ];then
        echo "start to predict......"

        CUDA_VISIBLE_DEVICES=2,3 python ../relation_classifier/relation_classify.py \
                                        --vocab_file cat_b24_lr1e5_t7_d03_relation_1223/vocab.txt    \
                                        --bert_config_file cat_b24_lr1e5_t7_d03_relation_1223/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.data.train \
                                        --eval_train_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                        --eval_file  /nas/jiangdanyang/projects/NLP-SubjectExtract-relation/src/data/res/figure_relation/dev.csv   \
                                        --train_batch_size  24  \
                                        --eval_batch_size 160 \
                                        --learning_rate 1e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 7   \
                                        --num_labels 2   \
                                        --result_file ./predict1223_7_24_dev_jdy.csv  \
                                        --output_dir  cat_b24_lr1e5_t7_d03_relation_1223  \
                                        --reduce_dim 768  \
                                        --gpu0_size 0  \
                                        --bert_model cat_b24_lr1e5_t7_d03_relation_1223  \
                                        --init_checkpoint cat_b24_lr1e5_t7_d03_relation_1223/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 12 3>&2 2>&1 1>&3 | tee logs/1223predict.log

elif [ "$1" = "predict-big" ];then
        echo "start to predict......"

         CUDA_VISIBLE_DEVICES=0,1,2,3 python ../relation_classifier/relation_classify.py \
                                        --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                        --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                        --do_lower_case    \
                                        --train_file /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.data.train \
                                        --eval_train_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/small/small.figure_relation_2281  \
                                        --eval_file  /nas/lishengping/datas/person2vector/figure_relation_lsp_final/dev/figure_relation_1    \
                                        --train_batch_size  24  \
                                        --eval_batch_size 80 \
                                        --learning_rate 1e-5   \
                                        --num_train_epochs 6   \
                                        --top_n 20   \
                                        --num_labels 2   \
                                        --result_file ./predict0701_7_20_dev.csv  \
                                        --output_dir  ./cat_b20_lr1e5_t7_d03_relation_0701  \
                                        --reduce_dim 0  \
                                        --gpu0_size 0  \
                                        --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                        --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                        --do_predict   \
                                        --gradient_accumulation_steps 12 3>&2 2>&1 1>&3 | tee logs/activate_cls_abs_model0525.log

else
    echo 'unknown argment 1'
fi
