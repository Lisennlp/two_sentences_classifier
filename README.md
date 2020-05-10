# 语义相似度，句向量生成

本项目可用于训练涉及到两句的分类任务，同时，可基于训练好的模型，获取句向量，用于下游相关任务，比如语义相似度任务。

涉及论文：《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》

#### 训练

    cd script
    sh train.sh

#### 预测

    cd script
    sh predict.sh

#### 获取句向量

    cd examples
    python embdeeings_examples.py

具体的做法可看two_sentences_classifier/train.py文件
