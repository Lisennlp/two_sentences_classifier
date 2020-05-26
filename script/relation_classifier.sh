# CUDA_VISIBLE_DEVICES=0,1 python ../two_sentences_classifier/three_categories_train2.py \
#                                   --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
#                                   --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
#                                   --do_lower_case    \
#                                   --train_file /nas/xd/data/novels/figure_relation/ \
#                                   --eval_file  /nas/xd/data/novels/figure_relation/data.dev   \
#                                   --train_batch_size 36   \
#                                   --eval_batch_size 36 \
#                                   --learning_rate 5e-5   \
#                                   --num_train_epochs 6   \
#                                   --top_n 7   \
#                                   --num_labels 4   \
#                                   --output_dir ./two_sentences_classifier_model0513_7  \
#                                   --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
#                                   --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
#                                   --do_train   \
#                                   --gradient_accumulation_steps 1 3>&2 2>&1 1>&3 | tee logs/two_sentences_classifier_7.log
CUDA_VISIBLE_DEVICES=3,4,5 python ../two_sentences_classifier/add_type_train.py \
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
                                  --output_dir ./384_2_activate_cls_abs_model0526 \
                                  --reduce_dim 384  \
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 5 3>&2 2>&1 1>&3 | tee logs/384_2_activate_cls_abs_model0526.log

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ../two_sentences_classifier/train.py \
#                                   --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
#                                   --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
#                                   --do_lower_case    \
#                                   --train_file //nas/xd/data/novels/figure_relation/ \
#                                   --eval_file  /nas/xd/data/novels/figure_relation/data.dev   \
#                                   --train_batch_size  32  \
#                                   --eval_batch_size 16 \
#                                   --learning_rate 5e-5   \
#                                   --num_train_epochs 6   \
#                                   --top_n 7   \
#                                   --num_labels 2   \
#                                   --output_dir ./origin_model0513_7  \
#                                   --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
#                                   --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
#                                   --do_train   \
#                                   --gradient_accumulation_steps 2 3>&2 2>&1 1>&3 | tee logs/origin_model0513_7.log

# # 预测
# CUDA_VISIBLE_DEVICES=5,6,7 python ../two_sentences_classifier/add_type_train.py \
#                                   --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
#                                   --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
#                                   --do_lower_case    \
#                                   --train_file /nas/xd/data/novels/figure_relation/data.train \
#                                   --eval_train_file  /nas/xd/data/novels/figure_relation/2001.train  \
#                                   --eval_file  /nas/xd/data/novels/figure_relation/data.dev   \
#                                   --train_batch_size 30   \
#                                   --eval_batch_size 5 \
#                                   --learning_rate 3e-5   \
#                                   --num_train_epochs 6   \
#                                   --top_n 7   \
#                                   --num_labels 2   \
#                                   --result_file ./add_type_model0515_7_1/predict0518_dev.csv  \
#                                   --output_dir ./add_type_model0515_7_1  \
#                                   --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
#                                   --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
#                                   --do_predict   \
#                                   --gradient_accumulation_steps 6 3>&2 2>&1 1>&3 | tee logs/add_type_model0515_7_1.log
