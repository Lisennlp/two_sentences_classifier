import sys
import os
import math
import pandas as pd

print(pd.__file__)

sys.path.append('../')

from two_sentences_classifier.word_embeddings import EmbeddingsModel, RelationModelEmbeddings

if __name__ == "__main__":
    # 人物对话分类词向量
    # model = EmbeddingsModel(
    #     '/nas/lishengping/two_classifier_models/two_sentences_classifier_model0427')
    # sentences = ['你要去干嘛', '你今天去公司吗', "我想去旅游", "五一都干嘛了", "明天又要上班了", "要去吃饭了", "你喜欢打篮球吗"]
    # sentences = sentences * 30
    # sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences)

    # 人物关系分类词向量
    model = RelationModelEmbeddings(
        '/nas/lishengping/relation_models/activate_cls_abs_model0531_15')
    sentences = [
        '你要去干嘛||你今天去公司吗', '你今天去公司吗||你今天去公司吗', "我想去旅游||你今天去公司吗", "五一都干嘛了||你今天去公司吗",
        "明天又要上班了||你今天去公司吗", "要去吃饭了||你今天去公司吗", "你喜欢打篮球吗||你今天去公司吗"
    ]
    sentences = sentences
    sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences, split='||')

    print(f'sentences_mean_vector = {sentences_mean_vector}')
    print(f'sentences_vector_modes = {sentences_vector_modes}')
