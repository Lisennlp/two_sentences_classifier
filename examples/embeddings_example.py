import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

from two_sentences_classifier.word_embeddings import EmbeddingsModel


if __name__ == "__main__":
    model = EmbeddingsModel(
        '/nas/lishengping/two_classifier_models/two_sentences_classifier_model0427')
    sentences = ['你要去干嘛', '你今天去公司吗', "我想去旅游", "五一都干嘛了", "明天又要上班了", "要去吃饭了", "么么哒！"]
    sentences = sentences * 10
    sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences)
    print(f'sentences_mean_vector = {sentences_mean_vector}')
    print(f'sentences_vector_modes = {sentences_vector_modes}')
    print(f'sentences_vector_modes len = {len(sentences_vector_modes)}')
