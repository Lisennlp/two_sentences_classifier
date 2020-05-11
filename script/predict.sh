#段落sigmoid打分
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python ../two_sentences_classifier/train.py \
                                 --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \
                                 --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                 --do_lower_case    \
                                 --train_file /nas/xd/data/novels/speech_labeled20/small.data.train  \
                                 --eval_file  /nas/xd/data/novels/speech_labeled20/data.dev \
                                 --train_batch_size 32   \
                                 --eval_batch_size 30   \
                                 --learning_rate 5e-5   \
                                 --num_train_epochs 6   \
                                 --top_n 20   \
                                 --num_labels 2   \
                                 --output_dir /home/lishengping/caiyun_projects/two_sentences_classifier/script/two_sentences_classifier_large_model0506_7  \
                                 --result_file ./predict_result0507_7_20.csv  \
                                 --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                 --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                 --do_predict   \
                                 --gradient_accumulation_steps 1 3>&2 2>&1 1>&3 | tee logs/two_classfier_predict0429.log