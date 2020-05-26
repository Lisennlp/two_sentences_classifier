"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from itertools import chain
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import classification_report
from collections import defaultdict
from parallel import BalancedDataParallel

import tokenization
from modeling import BertConfig, BertForSequenceClassification
from optimization import BertAdam
from data_untils import get_span, filter_lines, normalize, is_chapter_name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'bert_config.json'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, text_a, text_b=None, label=None, name=None, person=None):
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.name = name
        self.person = person


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 position_ids=None,
                 example_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_id = example_id
        self.position_ids = position_ids


class DataProcessor(object):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.map_symbols = {"’": "'", "‘": "'", "“": '"', "”": '"'}

    def read_novel_examples(self, path, top_n=5, task_name='train'):
        top_n = int(top_n)
        examples = []
        example_map_ids = {}
        zero, one, index = 0, 0, 0
        with open(path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f):
                line = line.replace('\n', '').split('\t')
                text_a = line[0].split('||')
                start = (len(text_a) - top_n) // 2
                end = len(text_a) - start
                indice = slice(start, end)
                text_a = text_a[indice]
                try:
                    assert len(text_a) == top_n
                except Exception as e:
                    print(f'error = {e}')
                    print(f'line = {line}')
                    print(f'text_a = {text_a}')
                    continue

                if int(line[-3]) not in [0, 1]:
                    print(f'line[-3] = {line[-3]}')
                    continue

                example = InputExample(id=index, text_a=text_a, label=int(line[-3]), name=line[-2])
                if int(line[-3]) == 1:
                    one += 1
                else:
                    zero += 1
                examples.append(example)
                if task_name != 'train':
                    example_map_ids[index] = example
                index += 1
        print(f'{os.path.split(path)[-1]} file examples {len(examples)}')
        print(f'one = {one}, zero = {zero}')
        return examples, example_map_ids

    def read_predict_examples(self, path, top_n=5, task_name='eval'):
        top_n = int(top_n)
        examples = []
        index = 0
        with open(path, 'r') as f:
            lines = filter_lines(f, keep_chapter_name=True)
            for i, new_line in enumerate(lines):
                if is_chapter_name(new_line):
                    continue
                speaker, speech = new_line.split('::', maxsplit=1)
                if not speaker == '旁白':
                    continue
                indexs = get_span(lines, i)
                if not indexs:
                    continue
                text_a = [normalize(lines[i]) for i in indexs]
                assert len(text_a) == top_n
                example = InputExample(id=index, text_a=text_a, label=i)
                examples.append(example)
                index += 1
        print(f'examples len = {len(examples)}')
        print(f'lines len = {len(lines)}')

        return examples, lines


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        sents_tokens, segment_ids = [], []
        for i, sent in enumerate(example.text_a):
            if i == 0:
                sent_tokens = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
            else:
                sent_tokens = tokenizer.tokenize(sent) + ['[SEP]']

            sents_tokens.extend(sent_tokens)
            segment_ids.extend([i] * len(sent_tokens))
        sents_tokens = sents_tokens[:-1][:max_seq_length - 1] + ['[SEP]']
        length = len(sents_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(sents_tokens)
        segment_ids = segment_ids[:length]
        input_masks = [1] * length

        assert len(input_ids) == len(input_masks) == len(segment_ids)
        while length < max_seq_length:
            input_ids.append(0)
            input_masks.append(0)
            segment_ids.append(0)
            length += 1

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_masks,
                          segment_ids=segment_ids,
                          label_id=example.label,
                          example_id=example.id))

    print(f'feature example input_ids：{features[-1].input_ids}')
    print(f'feature example input_mask：{features[-1].input_mask}')
    print(f'feature example segment_ids：{features[-1].segment_ids}')

    print(f'total features {len(features)}')
    return features


def init_model_token_type(model, type_vocab_size=7):
    type_vocab_size = int(type_vocab_size)
    config = model.config
    config.type_vocab_size = type_vocab_size
    token_type_embeddings_appended = torch.nn.Embedding(type_vocab_size, config.hidden_size)
    token_type_embeddings_appended.weight.data.normal_(mean=0.0, std=config.initializer_range)
    encoder_token_type_embedding = model.bert.embeddings.token_type_embeddings
    token_type_embeddings_appended.weight.data[:
                                               2, :] = encoder_token_type_embedding.weight.data[:, :]
    encoder_token_type_embedding.weight = token_type_embeddings_appended.weight


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The train file path")
    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The dev file path")
    parser.add_argument("--eval_train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The train  eval file path")
    parser.add_argument("--predict_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The predict file path")
    parser.add_argument("--top_n",
                        default=5,
                        type=float,
                        required=True,
                        help="higher than threshold is classify 1,")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                        "This specifies the model architecture.")
    parser.add_argument("--bert_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                        "This specifies the model architecture.")
    parser.add_argument("--result_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The result file that the BERT model was trained on.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    # Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text.")
    parser.add_argument("--max_seq_length",
                        default=150,
                        type=int,
                        help="maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--num_labels", default=1, type=int, help="mapping classify nums")
    parser.add_argument("--reduce_dim",
                        default=64,
                        type=int,
                        help="from hidden size to this dimensions, reduce dim")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float,
                        default=128,
                        help='Loss scale, positive power of 2 can improve fp16 convergence.')

    args = parser.parse_args()

    data_processor = DataProcessor(args.num_labels)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False    # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu,
                bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    print(f'args.train_batch_size = {args.train_batch_size}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    bert_config.reduce_dim = args.reduce_dim

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}"
            .format(args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(
            args.output_dir))

    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file,
                                           do_lower_case=args.do_lower_case)

    def prepare_data(args, task_name='train'):
        if task_name == 'train':
            file_path = args.train_file
        elif task_name == 'eval':
            file_path = args.eval_file
        elif task_name == 'train_eval':
            file_path = args.eval_train_file
        elif task_name == 'predict':
            file_path = args.predict_file

        if os.path.isdir(file_path):
            examples = data_processor.read_file_dir(file_path, top_n=args.top_n)
        else:
            if task_name == 'predict':
                examples, example_map_ids = data_processor.read_predict_examples(
                    file_path, top_n=args.top_n, task_name=task_name)
            else:
                examples, example_map_ids = data_processor.read_novel_examples(file_path,
                                                                               top_n=args.top_n,
                                                                               task_name=task_name)
        features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)

        if task_name in ['train', 'eval', 'train_eval', 'predict']:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            datas = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids,
                                  all_label_ids)
        else:
            datas = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids)

        if task_name == 'train':
            if args.local_rank == -1:
                data_sampler = RandomSampler(datas)
            else:
                data_sampler = DistributedSampler(datas)
            dataloader = DataLoader(datas,
                                    sampler=data_sampler,
                                    batch_size=args.train_batch_size,
                                    drop_last=True)
        else:
            dataloader = DataLoader(datas, batch_size=args.eval_batch_size, drop_last=True)
        return (dataloader, example_map_ids) if task_name != 'train' else dataloader

    def accuracy(example_ids, logits, probs=None, data_type='eval'):
        logits = logits.tolist()
        example_ids = example_ids.tolist()
        assert len(logits) == len(example_ids)
        classify_name = ['no_answer', 'yes_answer']
        labels, text_a, novel_names = [], [], []
        map_dicts = example_map_ids if data_type == 'eval' else train_example_map_ids
        for i in example_ids:
            example = map_dicts[i]
            labels.append(example.label)
            text_a.append("||".join(example.text_a))
            novel_names.append(example.name)

        write_data = pd.DataFrame({
            "text_a": text_a,
            "labels": labels,
            "logits": logits,
            "novel_names": novel_names,
        })
        write_data['yes_or_no'] = write_data['labels'] == write_data['logits']
        if probs is not None:
            write_data['logits'] = probs.tolist()
        write_data.to_csv(args.result_file, index=False)
        assert len(labels) == len(logits)
        result = classification_report(labels, logits, target_names=classify_name)
        return result

    def eval_model(model, eval_dataloader, device, data_type='eval'):
        model.eval()
        eval_loss = 0
        all_logits = []
        all_example_ids = []
        all_probs = []
        accuracy_result = None
        batch_count = 0
        for step, batch in enumerate(tqdm(eval_dataloader, desc="evaluating")):
            example_ids, input_ids, input_mask, segment_ids, label_ids = batch
            if not args.do_train:
                label_ids = None
            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, labels=label_ids)
                argmax_logits = torch.argmax(logits, dim=1)
                first_indices = torch.arange(argmax_logits.size()[0])
                logits_probs = logits[first_indices, argmax_logits]
            if args.do_train:
                eval_loss += tmp_eval_loss.mean().item()
                all_logits.append(argmax_logits)
                all_example_ids.append(example_ids)
            else:
                all_logits.append(argmax_logits)
                all_example_ids.append(example_ids)
                all_probs.append(logits_probs)
            batch_count += 1
        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_example_ids = torch.cat(all_example_ids, dim=0)
            all_probs = torch.cat(all_probs, dim=0) if len(all_probs) else None
            accuracy_result = accuracy(all_example_ids,
                                       all_logits,
                                       probs=all_probs,
                                       data_type=data_type)
        eval_loss /= batch_count
        print(f'========= {data_type} acc ============\n')
        print(f'{accuracy_result}')
        return eval_loss, accuracy_result, all_logits

    def predict_model(model, predict_dataloader, device):
        model.eval()
        all_label_ids = []
        all_probs = []
        scene_cut_indexs = []

        for step, batch in enumerate(tqdm(predict_dataloader, desc="predicting")):
            example_ids, input_ids, input_mask, segment_ids, label_ids = batch
            # print(f'input_ids = {input_ids.shape}')
            # print(f'input_mask = {input_mask.shape}')
            # print(f'segment_ids = {segment_ids.shape}')
            # print(f'label_ids = {label_ids.shape}')

            with torch.no_grad():
                _, logits = model(input_ids, segment_ids, input_mask, labels=None)
                argmax_logits = torch.argmax(logits, dim=1)
                first_indices = torch.arange(argmax_logits.size()[0])
                logits_probs = logits[first_indices, argmax_logits]
                all_label_ids.extend(label_ids.tolist())
                all_probs.extend(logits_probs.tolist())
                for label, prob, pred_label in zip(label_ids.tolist(), logits_probs.tolist(), argmax_logits.tolist()):
                    if pred_label:
                        scene_cut_indexs.append(label)

                    # print(f'prob = {prob}')
                    # print(f'pred label = {pred_label}')
                    # print(f'sent = {predict_example_map_ids[label]}')

        print(f'all_example_ids = {len(all_label_ids)}')
        print(f'all_probs = {len(all_probs)}')

        assert len(all_label_ids) == len(all_probs)
        return scene_cut_indexs

    if args.do_train:
        train_dataloader = prepare_data(args, task_name='train')
        num_train_steps = int(
            len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs)

    model = BertForSequenceClassification(bert_config, num_labels=data_processor.num_labels)
    new_state_dict = model.state_dict()
    init_state_dict = torch.load(os.path.join(args.bert_model, 'pytorch_model.bin'))
    for k, v in init_state_dict.items():
        if k in new_state_dict:
            print(f'k in = {k} v in shape = {v.shape}')
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    init_model_token_type(model, type_vocab_size=args.top_n)

    for k, v in model.state_dict().items():
        print(f'k = {k}, v shape {v.shape}')

    if args.fp16:
        model.half()

    if args.do_predict:
        model_path = os.path.join(args.output_dir, WEIGHTS_NAME)
        new_state_dict = torch.load(model_path)
        new_state_dict = dict([
            (k[7:], v) if k.startswith('module') else (k, v) for k, v in new_state_dict.items()
        ])
        model.load_state_dict(new_state_dict)

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not device == 'cpu':
        model = torch.nn.DataParallel(model)
    #     # model = BalancedDataParallel(1, model, dim=0).to(device)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if n not in no_decay],
        'weight_decay_rate': 0.01
    }, {
        'params': [p for n, p in param_optimizer if n in no_decay],
        'weight_decay_rate': 0.0
    }]

    # eval_dataloader, example_map_ids = prepare_data(args, task_name='eval')
    # train_eval_dataloader, train_example_map_ids = prepare_data(args, task_name='train_eval')

    if args.do_train:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)

        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        eval_loss, acc, _ = eval_model(model, eval_dataloader, device)
        logger.info(f'初始开发集loss: {eval_loss}')
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            torch.cuda.empty_cache()
            model_save_path = os.path.join(args.output_dir, f"{WEIGHTS_NAME}.{epoch}")
            tr_loss = 0
            train_batch_count = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="training")):
                _, input_ids, input_mask, segment_ids, label_ids = batch
                # label_ids = label_ids.to(device)
                # input_ids = input_ids.to(device)
                # input_mask = input_mask.to(device)
                # segment_ids = segment_ids.to(device)

                loss, _ = model(input_ids, segment_ids, input_mask, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                train_batch_count += 1
            tr_loss /= train_batch_count
            eval_loss, acc, _ = eval_model(model, eval_dataloader, device)
            eval_model(model, train_eval_dataloader, device, data_type='train_eval')
            logger.info(
                f'训练loss: {tr_loss}, 开发集loss：{eval_loss} 训练轮数：{epoch + 1}/{int(args.num_train_epochs)}'
            )
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model.state_dict(), model_save_path)
            if epoch == 0:
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

    if args.do_predict:
        # eval_model(model, train_eval_dataloader, device, data_type='train_eval')
        # for root, dir_, files in os.walk(''):
        #     for file in files:
        #         arg.predict_file = os.path.join(root, file)

        predict_dataloader, predict_example_map_ids = prepare_data(args, task_name='predict')
        scene_cut_indexs = predict_model(model, predict_dataloader, device)
        predict_novel_name = os.path.join(os.path.split(args.predict_file)[0], 'scene_cut_datas', os.path.split(args.predict_file)[-1])
        for i in scene_cut_indexs:
            predict_example_map_ids[i] = '########' + predict_example_map_ids[i]
        with open(predict_novel_name, 'w', encoding='utf-8') as f:
            for i, line in enumerate(predict_example_map_ids):
                f.write(line)


if __name__ == "__main__":
    main()
