"""Embeddings"""

import os
import sys
from itertools import chain
import random

import torch

sys.path.append("../common_file")

print(f'{sys.path}')

from modeling import TwoSentenceClassifier, BertConfig, RelationClassifier
import tokenization


class EmbeddingsModel(object):

    def __init__(self, model_path):
        """ to obtain sentences embeddings model
            model path: init model weight path
        """
        self.model_path = model_path
        self.init_modle(TwoSentenceClassifier)

    def init_modle(self, model):
        print(f'starting to init model')
        vocab_path = os.path.join(self.model_path, 'vocab.txt')
        bert_config_file = os.path.join(self.model_path, 'bert_config.json')
        self.bert_config = BertConfig.from_json_file(bert_config_file)
        self.model = model(self.bert_config, 2)
        weight_path = os.path.join(self.model_path, 'pytorch_model.bin')
        new_state_dict = torch.load(weight_path)
        new_state_dict = dict([
            (k[7:], v) if k.startswith('module') else (k, v) for k, v in new_state_dict.items()
        ])
        self.model.load_state_dict(new_state_dict)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path)
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        print(f'init {model}  model finished')

    def convert_examples_to_features(self, sentences: list, max_seq_length=150):
        """convert id to features"""
        all_input_ids, all_input_masks, all_segment_ids = [], [], []
        for (ex_index, sent) in enumerate(sentences):
            sent_tokens = ['[CLS]'] + self.tokenizer.tokenize(sent)[:max_seq_length - 2] + ['[SEP]']
            length = len(sent_tokens)
            sent_segment_ids = [0] * length
            sent_input_masks = [1] * length
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            while length < max_seq_length:
                sent_input_ids.append(0)
                sent_input_masks.append(0)
                sent_segment_ids.append(0)
                length += 1
            assert len(sent_segment_ids) == len(sent_input_ids) == len(sent_input_masks)
            all_input_ids.append(torch.tensor(sent_input_ids).view(1, -1))
            all_input_masks.append(torch.tensor(sent_input_masks).view(1, -1))
            all_segment_ids.append(torch.tensor(sent_segment_ids).view(1, -1))
        return all_input_ids, all_input_masks, all_segment_ids

    def embeddings(self, sentences: list, batch_size=30, **kwargs):
        """ 
            **kwargs：
            batch_size: one circle sentence numbers
            max_seq_length: max sentences length
            split：split symbol，if get split， to relation modeling features convert
        """
        all_input_ids, all_input_mask, all_segment_ids = self.convert_examples_to_features(
            sentences, **kwargs)
        output_vectors = []
        # print(f'all_input_ids = {len(all_input_ids)}')
        with torch.no_grad():
            for i in range(0, len(all_input_ids), batch_size):
                if i % batch_size == 0:
                    input_ids = torch.cat(all_input_ids[i:i + batch_size],
                                          dim=0).to(self.device).unsqueeze(0)
                    segment_ids = torch.cat(all_segment_ids[i:i + batch_size],
                                            dim=0).to(self.device).unsqueeze(0)
                    input_mask = torch.cat(all_input_mask[i:i + batch_size],
                                           dim=0).to(self.device).unsqueeze(0)
                    output_vector = self.model(input_ids, segment_ids, input_mask, embedding=True)
                    output_vectors.append(output_vector)
            output_vectors = torch.cat(output_vectors, dim=1).squeeze()
            # vector_mode: bsz
            sentences_vector_modes = torch.sqrt((output_vectors * output_vectors).sum(-1)).squeeze()
            sentences_mean_vector = output_vectors.mean(0).squeeze()
            assert len(sentences_mean_vector) == self.bert_config.hidden_size
        return sentences_mean_vector.tolist(), sentences_vector_modes.tolist()


class RelationModelEmbeddings(EmbeddingsModel):
    """
    realtion classfier's embeddings
    """

    def __init__(self, model_path):
        """ to obtain sentences embeddings model
            model path: init model weight path
        """
        self.model_path = model_path
        self.init_modle(RelationClassifier)

    def convert_examples_to_features(self, sentences: list, max_seq_length=150, split='||'):
        """convert id to features"""
        input_ids, input_masks, segment_ids = [], [], []
        for (ex_index, sent) in enumerate(sentences):
            sents = sent.split(split)
            if len(sents) != 2:
                continue
            sents[0] = sents[0][:120].replace('"', '')
            sents[1] = sents[1][:120].replace('"', '')
            sents_token = [self.tokenizer.tokenize(s) for s in sents]
            sent_segment_ids = [0] * (len(sents_token[0]) + 2) + [1] * (len(sents_token[1]) + 1)
            sents_token = sents_token[0] + ['[SEP]'] + sents_token[1]
            sents_token = sents_token[:max_seq_length - 2]
            sent_segment_ids = sent_segment_ids[:max_seq_length]
            sents_token = ['[CLS]'] + sents_token + ['[SEP]']
            length = len(sents_token)
            sent_input_masks = [1] * length
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sents_token)

            while length < max_seq_length:
                sent_input_ids.append(0)
                sent_input_masks.append(0)
                sent_segment_ids.append(0)
                length += 1

            assert len(sent_segment_ids) == len(sent_input_ids) == len(sent_input_masks)
            input_ids.append(torch.tensor(sent_input_ids).view(1, -1))
            input_masks.append(torch.tensor(sent_input_masks).view(1, -1))
            segment_ids.append(torch.tensor(sent_segment_ids).view(1, -1))
        return input_ids, input_masks, segment_ids

    def classifiy(self, sentences: list, chunk_nums=7, split='||', **kwargs):
        """
        sentences: [[a||b, a||b, .....], [a||c, a||c, .....]] 或者 [a||b, ......, a||c]
        sent_nums: sentence numbers
        """
        if isinstance(sentences[0], list):
            sentences = chain(*sentences)

        # assert len(sentences) == sent_nums, 'sentence list length must equal to sent_nums'
        input_ids, input_mask, segment_ids = [
            torch.cat(i).unsqueeze(0).to(self.device)
            for i in self.convert_examples_to_features(sentences, split=split, **kwargs)
        ]
        print(f'in shape = {input_ids.shape}')
        with torch.no_grad():
            output_vectors, logits = self.model(input_ids, segment_ids, input_mask, embedding=False, chunk_nums=chunk_nums)
            pred_label = torch.argmax(logits)
            sentences_vector_modes = torch.sqrt((output_vectors * output_vectors).sum(-1)).squeeze()
            return sentences_vector_modes, pred_label


# if __name__ == "__main__":
#     # 人物对话分类词向量
#     # model = EmbeddingsModel(
#     #     '/nas/lishengping/two_classifier_models/two_sentences_classifier_model0427')
#     # sentences = ['你要去干嘛', '你今天去公司吗', "我想去旅游", "五一都干嘛了", "明天又要上班了", "要去吃饭了", "你喜欢打篮球吗"]
#     # sentences = sentences * 30
#     # sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences)

#     # 人物关系分类词向量
#     model = RelationModelEmbeddings(
#         '/nas/lishengping/relation_models/activate_cls_abs_model0531_15')
#     sentences = [
#         '你要去干嘛||你今天去公司吗', '你今天去公司吗||你今天去公司吗', "我想去旅游||你今天去公司吗", "五一都干嘛了||你今天去公司吗",
#         "明天又要上班了||你今天去公司吗", "要去吃饭了||你今天去公司吗", "你喜欢打篮球吗||你今天去公司吗"
#     ]
#     sentences = sentences
#     sentences_mean_vector, sentences_vector_modes = model.embeddings(sentences, split='||')

#     print(f'sentences_mean_vector = {sentences_mean_vector}')
#     print(f'sentences_vector_modes = {sentences_vector_modes}')
