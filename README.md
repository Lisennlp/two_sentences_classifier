# 语义相似度，句向量生成

本项目可用于训练涉及到两句的分类任务，同时，可基于训练好的模型，获取句向量，用于下游相关任务，比如语义相似度任务。

涉及论文：《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》

#### 训练

    cd script
    sh train.sh

    - 参数讲解
    
    CUDA_VISIBLE_DEVICES=4,5,6,7 python ../two_sentences_classifier/train.py \
                                  --vocab_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/vocab.txt    \  # 词表文件，这里用的是roberta
                                  --bert_config_file /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/bert_config.json   \
                                  --do_lower_case    \
                                  --train_file /nas/xd/data/novels/speech_labeled20/  \   # 训练文件目录
                                  --eval_file  /nas/xd/data/novels/speech_labeled20/data.dev   \   # 开发集文件
                                  --train_batch_size 32   \
                                  --eval_batch_size 8 \
                                  --learning_rate 5e-5   \
                                  --num_train_epochs 6   \
                                  --top_n 15   \   # 每天数据有20句，取了top_n句
                                  --num_labels 2   \  # 类别数目
                                  --output_dir ./two_sentences_classifier_large_model0506_15   \  # 模型保存位置
                                  --bert_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/   \
                                  --init_checkpoint /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/pytorch_model.bin   \
                                  --do_train   \
                                  --gradient_accumulation_steps 4 3>&2 2>&1 1>&3 | tee logs/two_sentences_classifier_large_model0430_15.log


#### 框架图
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020052215541041.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzkyMjkwMQ==,size_16,color_FFFFFF,t_70)
    


    
#### 预测

    cd script
    sh predict.sh

#### 获取句向量

    cd examples
    python embdeeings_examples.py

具体的做法可看two_sentences_classifier/train.py文件
