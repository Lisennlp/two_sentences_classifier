
import sys
import os
import math

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))


import pandas as pd
from two_sentences_classifier.word_embeddings import EmbeddingsModel, RelationModelEmbeddings

if __name__ == "__main__":
    path = '/nas/xd/data/novels/figure_relation/results/predict0528_dev_7_7.csv'
    path = '/nas/xd/data/novels/figure_relation/results//predict0602_15_20_dev.csv'
    model = RelationModelEmbeddings(
        '/nas/lishengping/relation_models/activate_cls_abs_model0531_15')
    data = pd.read_csv(path)
    writer_file = open('/nas/xd/data/novels/figure_relation/results/predict0602_15_20_dev_emb',
                       'w',
                       encoding='utf-8')

    for i, (text_a, text_b, label, logit, novel_name, yes_or_no, person) in enumerate(
            zip(data['text_a'], data['text_b'], data['labels'], data['logits'], data['novel_names'],
                data['yes_or_no'], data['person'])):
        text_a_emb = text_a.split('||')
        text_b_emb = text_b.split('||')

        _, a_vector_modes = model.embeddings(text_a_emb, split='""')
        _, b_vector_modes = model.embeddings(text_b_emb, split='""')
        if i % 10 == 0:
            print(f"当前进度: {i}/{len(data['person'])}")
        writer_str = f'{text_a}\t{text_b}\t{label}\t{novel_name}\t{logit}\t{yes_or_no}\t{person}\t{a_vector_modes}\t{b_vector_modes}\n'
        writer_file.write(writer_str)

    writer_file.close()
