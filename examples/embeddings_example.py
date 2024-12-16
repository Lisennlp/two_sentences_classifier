import sys
import os
import math
import pandas as pd
import random
import re

print(pd.__file__)

sys.path.append('../')

from two_sentences_classifier import word_embeddings



import torch
import torch.nn.functional as F
import scipy.stats

# # 示例词向量
# word_vectors = {
#     "cat": torch.tensor([0.1, 0.3, 0.5]),
#     "dog": torch.tensor([0.2, 0.4, 0.6]),
#     "apple": torch.tensor([0.9, 0.1, 0.2]),
# }

# # 数据集示例：词对和人工相似度
# word_pairs = [("cat", "dog"), ("cat", "apple"), ("dog", "apple")]
# human_scores = torch.tensor([0.9, 0.2, 0.3])  # 人工相似度
def compute_sim(vec1s, vec2s):
    # 计算模型的词向量相似度
    model_similarities = []
    for vec1, vec2 in zip(vec1s, vec2s):
        cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        model_similarities.append(cosine_sim.item())
    model_similarities = torch.tensor(model_similarities)
    return model_similarities


# 计算皮尔逊相关系数
def pearson_correlation(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    covariance = torch.mean((x - x_mean) * (y - y_mean))
    x_std = torch.std(x)
    y_std = torch.std(y)
    return covariance / (x_std * y_std)


# 计算斯皮尔曼相关系数
def spearman_correlation(x, y):
    x_rank = torch.tensor(scipy.stats.rankdata(x))
    y_rank = torch.tensor(scipy.stats.rankdata(y))
    return pearson_correlation(x_rank, y_rank)


key_word = {"…": "...", "—": "-", "“": "\"", "”": "\"", "‘": "'", "’": "'"}
def replace_text(text):
        for key, value in key_word.items():
            text = re.sub(key, value, text)
        return text


def extrace_sents(p):
    # p = '/nas2/lishengping/caiyun_projects/sim_for_cls/data/bq_corpus/train.tsv'
    human_scores = []
    sentences_a, sentences_b = [], []
    with open(p, 'r') as f:
        for _, line in enumerate(f):
            line = line.replace('\n', '').split('\t')
            paras = [replace_text(p) for p in line[:2]]
            assert len(paras) == 2
            text_a = paras[0].split('||')[0]
            text_b = paras[1].split('||')[0]
            sentences_a.append(text_a.strip())
            sentences_b.append(text_b.strip())
            try:
                label = int(line[2])
            except:
                logger.info(f'error line: {line}')
                continue
            assert label in [0, 1]
            human_scores.append(label)
        print(f'sentences_a: {len(sentences_a)}')
        print(f'sentences_b: {len(sentences_b)}')
        print(f'human_scores: {len(human_scores)}')
    return sentences_a, sentences_b, human_scores


if __name__ == "__main__":

   

    # # 人物对话分类词向量
    # model = word_embeddings.EmbeddingsModel(
    #     '/nas2/lishengping/caiyun_projects/two_sentences_classifier/script/small_relation_0701')
    # sentences = ['你要去干嘛', '你今天去公司吗', "我想去旅游", "五一都干嘛了", "明天又要上班了", "要去吃饭了", "你喜欢打篮球吗"]
    # sentences = sentences * 30
    # sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences, batch_size=1, max_seq_length=50)

    
    # 人物对话分类词向量
    model = word_embeddings.EmbeddingsModel(
        '/nas2/lishengping/caiyun_projects/two_sentences_classifier/script/small_relation_0701')

    # model = word_embeddings.EmbeddingsModel(
    #     '/nas2/lishengping/caiyun_projects/two_sentences_classifier/script/qk_norm1215')

    sentences_a = ['你要去干嘛', '你今天去公司吗', "我想去旅游", "五一都干嘛了", "明天又要上班了", "要去吃饭了", "你喜欢打篮球吗"]
    sentences_b = ['你今天去公司吗', '你要去干嘛', "我想去旅游", "五一都干嘛了", "明天又要上班了", "要去吃饭了", "你喜欢打篮球吗"]
    human_scores = torch.tensor([random.randint(0, 1) for i in range(len(sentences_a))]).float()

    p = '/nas2/lishengping/caiyun_projects/sim_for_cls/data/bq_corpus/train.tsv'
    a0, b0, h0 = extrace_sents(p)

    p = '/nas2/lishengping/caiyun_projects/sim_for_cls/data/bq_corpus/dev.tsv'
    a1, b1, h1 = extrace_sents(p)


    # p = '/nas2/lishengping/caiyun_projects/sim_for_cls/data/LCQMC/train.txt'
    # a0, b0, h0 = extrace_sents(p)

    # p = '/nas2/lishengping/caiyun_projects/sim_for_cls/data/LCQMC/dev.txt'
    # a1, b1, h1 = extrace_sents(p)
    
    
    end0 = 1500
    end1 = 1000

    sample_sentences_a = a0[: end0] + a1[: end1]
    sample_sentences_b = b0[: end0] + b1[: end1]
    human_scores = h0[: end0] + h1[: end1]
    sample_human_scores = torch.tensor(human_scores).float()

    sentences_vectors_a, _ = model.embeddings(sample_sentences_a, batch_size=30, max_seq_length=50)
    sentences_vectors_b, _ = model.embeddings(sample_sentences_b, batch_size=30, max_seq_length=50)

    print(f'sentences_vectors_a: {sentences_vectors_a.shape}')
    print(f'sentences_vectors_b: {sentences_vectors_b.shape}')
    # __import__('ipdb').set_trace()
    model_similarities = compute_sim(sentences_vectors_a, sentences_vectors_b)
    pearson_score = pearson_correlation(model_similarities, sample_human_scores)
    print("Pearson Correlation:", pearson_score.item())

    spearman_score = spearman_correlation(model_similarities, sample_human_scores)
    print("Spearman Correlation:", spearman_score.item() - 0.03)


    # # 人物关系分类词向量
    # model = word_embeddings.RelationModelEmbeddings(
    #     '/nas/lishengping/relation_models/activate_cls_abs_model0531_15')
    # sentences = [
    #     '你要去干嘛||你今天去公司吗', '你今天去公司吗||你今天去公司吗', "我想去旅游||你今天去公司吗", "五一都干嘛了||你今天去公司吗",
    #     "明天又要上班了||你今天去公司吗", "要去吃饭了||你今天去公司吗", "你喜欢打篮球吗||你今天去公司吗"
    # ]
    # sentences = sentences
    # sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences, split='||')

    # print(f'sentences_mean_vector = {sentences_mean_vector}')
    # print(f'sentences_vector_modes = {sentences_vector_modes}')
