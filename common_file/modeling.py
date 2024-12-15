# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'
WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'bert_config.json'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                weights = re.split(r'_(\d+)', m_name)
            else:
                weights = [m_name]
            if weights[0] == 'kernel' or weights[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif weights[0] == 'output_bias' or weights[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif weights[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif weights[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, weights[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(weights) >= 2:
                num = int(weights[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file,
                      str) or (sys.version_info[0] == 2
                               and isinstance(vocab_size_or_config_json_file, str)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):

        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):

        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                             "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # bsz x len x 8 x 64
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)    # bsz x 8 x len x 64

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)    # bsz x len x 512
        mixed_key_layer = self.key(hidden_states)    # bsz x len x 512
        mixed_value_layer = self.value(hidden_states)    # bsz x len x 512

        query_layer = self.transpose_for_scores(mixed_query_layer)    # bsz x 8 x len x 64
        key_layer = self.transpose_for_scores(mixed_key_layer)    # bsz x 8 x len x 64
        value_layer = self.transpose_for_scores(mixed_value_layer)    # bsz x 8 x len x 64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))    # bsz x 8 x len x len
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)    # bsz x 8 x len x len
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask    # mask部分是一个很小的数字，因此计算softmax之后几乎为0

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)    # bsz x 8 x len x 64
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()    # bsz x len x 8 x 64
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    """
     LayerNorm加入残差归一化
    """

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2
                                                  and isinstance(config.hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]    # 或者用ACT2FN里面定义的几个激活函数
        else:
            self.intermediate_act_fn = config.hidden_act    # 全连接层之后一般接激活函数，bert用gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    降维至768，残差归一化
    """

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        # first_token_tensor = hidden_states.max(1)[0]
        # pooled_output = self.dense(first_token_tensor)
        pooled_output = first_token_tensor
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2
                                                  and isinstance(config.hidden_act, str)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Model name '{}' was not found in model name list ({}). "
                         "We assumed '{}' was a path or url but couldn't find any file "
                         "associated to this path or url.".format(
                             pretrained_model_name_or_path,
                             ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys,
                                         unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)    # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]    # 最后一层的隐层状态
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                masked_lm_labels=None,
                next_sentence_label=None):
        # pooled_output是最后一层的CLS的隐层状态经过  768 * 768 -> tanh -> 768 * 768， 而sequence_output为最后一层隐层状态
        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                   attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                next_sentence_label=None):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class Expert(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.w0 = nn.Linear(config.hidden_size, intermediate_size)
        self.w1 = nn.Linear(config.hidden_size, intermediate_size)
        self.activ_fn = nn.SiLU()
        self.wo = nn.Linear(intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        w0 = self.w0(hidden_states)
        w1 = self.w1(hidden_states)
        gate = self.activ_fn(w0)
        intermediate = gate * w1
        hidden_states = self.wo(intermediate)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = 8
        self.top_k = 2
        self.norm_topk_prob = True
        shared_expert_intermediate_size = 512
        intermediate_size = 512
        self.gate = nn.Linear(config.hidden_size, 8, bias=False)
        self.experts = nn.ModuleList(
            [Expert(config, intermediate_size=intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Expert(config, intermediate_size=shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def euclidean_distance(vector_a, vector_b):
    vector_a = torch.tensor(vector_a, dtype=torch.float32)
    vector_b = torch.tensor(vector_b, dtype=torch.float32)
    distance = torch.sqrt(torch.sum((vector_a - vector_b) ** 2))
    return distance


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, init_alpha=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, output1, output2, label):
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-8)  # 加 1e-8 避免 sqrt(0)
        positive_loss = label * euclidean_distance
        negative_loss = (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0.0)
        alpha = torch.nn.functional.softplus(self.alpha)  # 约束为正值
        loss = torch.mean(positive_loss + negative_loss)
        return alpha * loss


class TwoSentenceClassifier(BertPreTrainedModel):
    """
        all token mean -> 拼接  -> dropout0.3 -> 分类
    """

    def __init__(self, config, num_labels, moe=False, os_loss=False):
        super(TwoSentenceClassifier, self).__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.3)
        if self.config.reduce_dim > 0:
            self.reduce_dimension = nn.Linear(3 * config.hidden_size, config.reduce_dim)
            self.classifier = nn.Linear(config.reduce_dim, num_labels)
        else:
            self.classifier = nn.Linear(3 * config.hidden_size, num_labels)
        self.activate = nn.Tanh()

        self.moe = moe
        self.os_loss = os_loss
        if self.moe:
            self.moe = MOE(config=config)

        if self.os_loss:
            self.os_distance = ContrastiveLoss()
        # 初始化参数
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                position_ids=None,
                labels=None,
                embedding=False):
        input_ids_size = input_ids.size()
        # bsz x 2 x len -> bsz*2 x len
        input_ids = input_ids.view(input_ids_size[0] * input_ids_size[1], input_ids_size[2])
        attention_mask = attention_mask.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        token_type_ids = token_type_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        if position_ids is not None:
            position_ids = position_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        # bsz x 1 -> bsz
        if labels is not None:
            labels = labels.view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.MultiMarginLoss()
        # sequence_output: bsz*14 x len x 768,  attention_mask:  bsz*14 x len
        all_sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       position_ids,
                                       output_all_encoded_layers=False)
        if self.moe:
            sequence_output, _ = self.moe(all_sequence_output)
        else:
            sequence_output = all_sequence_output
        # sequence_output = (all_sequence_output[-1] + all_sequence_output[-2]) / 2

        # bsz*14 x len -> bsz*14 x len x 768
        # print(f'attention_mask: {attention_mask.shape} sequence_output: {sequence_output.shape} all_sequence_output: {all_sequence_output.shape}')
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            sequence_output.size()).float()
        # bsz*14 x len x 768
        mask_sequence_output = sequence_output * attention_mask_expanded
        # -> bsz*14 x 1 x 768
        sum_mask_sequence_output = mask_sequence_output.sum(1)
        # bsz*14 x len x 768 -> bsz*14 x 1 x 768
        sum_attention_mask_expanded = attention_mask_expanded.sum(1)
        sum_attention_mask_expanded = torch.clamp(sum_attention_mask_expanded, min=1e-9)  # 除数不能为0
        # output_vectors: bsz*14 x 1 x 768
        output_vectors = sum_mask_sequence_output / sum_attention_mask_expanded
        # -> bsz x 14 x 768
        output_vectors = output_vectors.view(input_ids_size[0], -1, output_vectors.size()[-1])
        # -> 2个bsz x 7 x 768
        if embedding:
            return output_vectors
        sentence_a_vector, sentence_b_vector = torch.chunk(output_vectors, 2, dim=1)
        # -> 2个bsz x 768
        # print(f'sentence_a_vector = {sentence_a_vector.shape}')
        sentence_a_vector = sentence_a_vector.mean(1).squeeze(1)
        sentence_b_vector = sentence_b_vector.mean(1).squeeze(1)
        # 欧氏距离loss
        # bsz*num_labels x 768  -> # bsz*num_labels x 1
        # sentence_all_vector: bsz x 3*768
        sentence_c_vector = torch.abs(sentence_a_vector - sentence_b_vector)
        sentence_all_vector = torch.cat([sentence_a_vector, sentence_b_vector, sentence_c_vector],
                                        dim=1)
        if self.config.reduce_dim > 0:
            sentence_all_vector = self.reduce_dimension(sentence_all_vector)    # 降维
        sentence_all_vector = self.activate(sentence_all_vector)
        sentence_all_vector = self.dropout(sentence_all_vector)
        # bsz x 2
        logits = self.classifier(sentence_all_vector)
        if labels is not None:
            loss = loss_fn(logits, labels)
            if self.os_loss:
                os_distance_loss = self.os_distance(sentence_a_vector, sentence_b_vector, labels)
                return loss + os_distance_loss, os_distance_loss, logits
            else:
                return loss, logits
        else:
            return _, torch.softmax(logits, dim=-1)


class DynamicSbert(BertPreTrainedModel):
    """
        all token mean -> 拼接  -> dropout0.3 -> 分类
    """

    def __init__(self, config, num_labels, moe=False, os_loss=False):
        super(TwoSentenceClassifier, self).__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.3)
        if self.config.reduce_dim > 0:
            self.reduce_dimension = nn.Linear(3 * config.hidden_size, config.reduce_dim)
            self.classifier = nn.Linear(config.reduce_dim, num_labels)
        else:
            self.classifier = nn.Linear(3 * config.hidden_size, num_labels)
        self.activate = nn.Tanh()

        self.moe = moe
        self.os_loss = os_loss
        if self.moe:
            self.moe = MOE(config=config)

        if self.os_loss:
            self.os_distance = ContrastiveLoss()
        # 初始化参数
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                position_ids=None,
                labels=None,
                embedding=False,
                return_os_loss=False):
        input_ids_size = input_ids.size()
        # bsz x 2 x len -> bsz*2 x len
        input_ids = input_ids.view(input_ids_size[0] * input_ids_size[1], input_ids_size[2])
        attention_mask = attention_mask.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        token_type_ids = token_type_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        if position_ids is not None:
            position_ids = position_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        # bsz x 1 -> bsz
        if labels is not None:
            labels = labels.view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.MultiMarginLoss()
        # sequence_output: bsz*14 x len x 768,  attention_mask:  bsz*14 x len
        all_sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       position_ids,
                                       output_all_encoded_layers=False)
        if self.moe:
            sequence_output, _ = self.moe(all_sequence_output)
        else:
            sequence_output = all_sequence_output
        # sequence_output = (all_sequence_output[-1] + all_sequence_output[-2]) / 2

        # bsz*14 x len -> bsz*14 x len x 768
        # print(f'attention_mask: {attention_mask.shape} sequence_output: {sequence_output.shape} all_sequence_output: {all_sequence_output.shape}')
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            sequence_output.size()).float()
        # bsz*14 x len x 768
        mask_sequence_output = sequence_output * attention_mask_expanded
        # -> bsz*14 x 1 x 768
        sum_mask_sequence_output = mask_sequence_output.sum(1)
        # bsz*14 x len x 768 -> bsz*14 x 1 x 768
        sum_attention_mask_expanded = attention_mask_expanded.sum(1)
        sum_attention_mask_expanded = torch.clamp(sum_attention_mask_expanded, min=1e-9)  # 除数不能为0
        # output_vectors: bsz*14 x 1 x 768
        output_vectors = sum_mask_sequence_output / sum_attention_mask_expanded
        # -> bsz x 14 x 768
        output_vectors = output_vectors.view(input_ids_size[0], -1, output_vectors.size()[-1])
        # -> 2个bsz x 7 x 768
        if embedding:
            return output_vectors
        sentence_a_vector, sentence_b_vector = torch.chunk(output_vectors, 2, dim=1)
        # -> 2个bsz x 768
        # print(f'sentence_a_vector = {sentence_a_vector.shape}')
        sentence_a_vector = sentence_a_vector.mean(1).squeeze(1)
        sentence_b_vector = sentence_b_vector.mean(1).squeeze(1)
        # 欧氏距离loss
        # bsz*num_labels x 768  -> # bsz*num_labels x 1
        # sentence_all_vector: bsz x 3*768
        sentence_c_vector = torch.abs(sentence_a_vector - sentence_b_vector)
        sentence_all_vector = torch.cat([sentence_a_vector, sentence_b_vector, sentence_c_vector],
                                        dim=1)
        if self.config.reduce_dim > 0:
            sentence_all_vector = self.reduce_dimension(sentence_all_vector)    # 降维
        sentence_all_vector = self.activate(sentence_all_vector)
        sentence_all_vector = self.dropout(sentence_all_vector)
        # bsz x 2
        logits = self.classifier(sentence_all_vector)
        if labels is not None:
            loss = loss_fn(logits, labels)
            if self.os_loss:
                os_distance_loss = self.os_distance(sentence_a_vector, sentence_b_vector, labels)
                return loss + os_distance_loss, os_distance_loss, logits
            else:
                return loss, logits
        else:
            return _, torch.softmax(logits, dim=-1)


class RelationClassifier(BertPreTrainedModel):
    """
        cls mean -> 拼接 -> 降维768 -> tanh -> dropout 0.3 -> 分类
    """

    def __init__(self, config, num_labels):
        super(RelationClassifier, self).__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.reduce_dim, num_labels)
        self.activation = nn.Tanh()
        self.reduce_dimension = nn.Linear(3 * config.hidden_size, config.reduce_dim)
        # self.norm_layer = BertLayerNorm(config.hidden_size)

        # 初始化参数
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                position_ids=None,
                labels=None,
                **kwargs):
        # bsz x 14
        input_ids_size = input_ids.size()
        # bsz x 2 x len -> bsz*2 x len
        input_ids = input_ids.view(input_ids_size[0] * input_ids_size[1], input_ids_size[2])
        attention_mask = attention_mask.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        token_type_ids = token_type_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        if position_ids is not None:
            position_ids = position_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        # bsz x 1 -> bsz
        if labels is not None:
            labels = labels.view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        # sequence_output: bsz*14 x len x 768,  attention_mask:  bsz*14 x len
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       position_ids,
                                       output_all_encoded_layers=False)

        if kwargs.get("token_mean"):
            # bsz*num_labels x 768
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
                sequence_output.size()).float()
            mask_sequence_output = sequence_output * attention_mask_expanded
            # if kwargs.get("attention"):
            #     # bsz*14 x len x 1
            #     sequence_token_mode = torch.norm(mask_sequence_output, dim=-1)   
            sum_mask_sequence_output = mask_sequence_output.sum(1)
            sum_attention_mask_expanded = attention_mask_expanded.sum(1)
            sum_attention_mask_expanded = torch.clamp(sum_attention_mask_expanded, min=1e-9)
            # output_vectors: bsz*14 x 1 x 768
            output_vectors = sum_mask_sequence_output / sum_attention_mask_expanded
            if kwargs.get("attention"):
                return mask_sequence_output, output_vectors
        else:
            # CLS token embeddings: bsz*14 x 1 x 768
            output_vectors = sequence_output[:, 0, :]
            if kwargs.get("attention"):
                return output_vectors

        # -> bsz x 14 x 768
        output_vectors = output_vectors.view(input_ids_size[0], -1, output_vectors.size()[-1])

        if kwargs.get("embeddings"):
            return output_vectors

        chunk_nums = kwargs.get("chunk_nums") if kwargs.get("chunk_nums") else 7

        # -> bsz x 7 x 768
        # sentence_a_vector, sentence_b_vector = torch.chunk(output_vectors, 2, dim=1)
        sentence_a_vector = output_vectors[:, :chunk_nums, :]
        sentence_b_vector = output_vectors[:, chunk_nums:, :]

        # -> 2个bsz x 768
        sentence_a_vector = sentence_a_vector.mean(1).squeeze(1)
        sentence_b_vector = sentence_b_vector.mean(1).squeeze(1)
        # bsz*num_labels x 768  -> # bsz*num_labels x 1
        sentence_c_vector = torch.abs(sentence_a_vector - sentence_b_vector)
        # sentence_all_vector: bsz x 3*768
        sentence_all_vector = torch.cat([sentence_a_vector, sentence_b_vector, sentence_c_vector],
                                        dim=1)
        sentence_all_vector = self.activation(self.reduce_dimension(sentence_all_vector))
        sentence_all_vector = self.dropout(sentence_all_vector)
        # bsz x 2
        logits = self.classifier(sentence_all_vector)
        if labels is not None:
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return output_vectors, torch.softmax(logits, dim=-1)


class ThreeCategoriesClassifier(BertPreTrainedModel):
    """
        the pursose is to get 2 categories -> [0, 1], but when get 1 label, it alse has 2 subset categories. that is:
        0 ->  0
        1 -> 0, 1
        but we use 4 classifier to do it.
    """

    def __init__(self, config, num_labels):
        super(ThreeCategoriesClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob + 0.2)
        self.classifier = nn.Linear(3 * config.hidden_size, num_labels)
        self.activate = nn.Tanh()

        # 初始化参数
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                position_ids=None,
                labels=None,
                embedding=False):
        input_ids_size = input_ids.size()
        # bsz x 2 x len -> bsz*2 x len
        input_ids = input_ids.view(input_ids_size[0] * input_ids_size[1], input_ids_size[2])
        attention_mask = attention_mask.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        token_type_ids = token_type_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        if position_ids is not None:
            position_ids = position_ids.view(input_ids_size[0] * input_ids_size[1],
                                             input_ids_size[2])
        # bsz x 1 -> bsz
        if labels is not None:
            labels = labels.view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        # sequence_output: bsz*14 x len x 768,  attention_mask:  bsz*14 x len
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       position_ids,
                                       output_all_encoded_layers=False)
        # bsz*num_labels x 768
        # print(sequence_output.shape)  # 70 x 768
        # print(attention_mask.shape) # 70 x 145

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            sequence_output.size()).float()
        mask_sequence_output = sequence_output * attention_mask_expanded
        sum_mask_sequence_output = mask_sequence_output.sum(1)
        sum_attention_mask_expanded = attention_mask_expanded.sum(1)
        sum_attention_mask_expanded = torch.clamp(sum_attention_mask_expanded, min=1e-9)
        # output_vectors: bsz*14 x 1 x 768
        output_vectors = sum_mask_sequence_output / sum_attention_mask_expanded
        # -> bsz x 14 x 768
        output_vectors = output_vectors.view(input_ids_size[0], -1, output_vectors.size()[-1])
        # -> 2个bsz x 7 x 768
        if embedding:
            return output_vectors

        sentence_a_vector, sentence_b_vector = torch.chunk(output_vectors, 2, dim=1)
        # -> 2个bsz x 768
        # print(f'sentence_a_vector = {sentence_a_vector.shape}')
        sentence_a_vector = sentence_a_vector.mean(1).squeeze(1)
        sentence_b_vector = sentence_b_vector.mean(1).squeeze(1)
        # bsz*num_labels x 768  -> # bsz*num_labels x 1
        sentence_c_vector = torch.abs(sentence_a_vector - sentence_b_vector)
        # sentence_all_vector: bsz x 3*768
        sentence_all_vector = torch.cat([sentence_a_vector, sentence_b_vector, sentence_c_vector],
                                        dim=1)
        sentence_all_vector = self.activate(sentence_all_vector)
        sentence_all_vector = self.dropout(sentence_all_vector)
        # bsz x 4
        logits = self.classifier(sentence_all_vector)
        first_logits, sencond_logits = torch.chunk(logits, 2, dim=-1)
        # print(f'sencond_logits = {sencond_logits}')
        first_labels = torch.clamp(labels, max=1)
        second_labels = labels[first_labels == 1]
        sencond_logits_label = sencond_logits[first_labels == 1]

        if labels is not None:
            if second_labels.sum():
                second_labels -= 1
                loss = loss_fn(first_logits,
                               first_labels) + 0.3 * loss_fn(sencond_logits_label, second_labels)
            else:
                loss = loss_fn(first_logits, first_labels)
            return loss, (first_logits.softmax(-1), sencond_logits.softmax(-1))
        else:
            return _, torch.softmax(logits, dim=-1)


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob + 0.2)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # 初始化参数
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                position_ids=None):
        # 取得是bert解码层最后一层的第一个token的向量
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False,
                                     position_ids=position_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return _, torch.softmax(logits, dim=-1)
