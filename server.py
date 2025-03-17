import sys
import os
from itertools import chain
import json

# 修改路径1
sys.path.append('/nas2/lishengping/caiyun_projects/two_sentences_classifier/common_file/')

import torch
from flask import Flask, request, jsonify

import tokenization
from modeling import BertConfig, TwoSentenceClassifier


app = Flask(__name__)


def convert_text_to_ids(text_a, text_b):
    features = []
    max_seq_length = 50
    input_ids, input_masks, segment_ids = [], [], []
    for i, sent in enumerate(chain(text_a, text_b)):
        sent_length = len(sent)
        sents_token = tokenizer.tokenize(sent)
        sents_token = ['[CLS]'] + sents_token[:max_seq_length - 2] + ['[SEP]']
        sent_segment_ids = [0] * len(sents_token)
        length = len(sents_token)
        sent_input_masks = [1] * length
        sent_input_ids = tokenizer.convert_tokens_to_ids(sents_token)

        while length < max_seq_length:
            sent_input_ids.append(0)
            sent_input_masks.append(0)
            sent_segment_ids.append(0)
            length += 1

        assert len(sent_segment_ids) == len(sent_input_ids) == len(sent_input_masks)
        input_ids.append(sent_input_ids)
        input_masks.append(sent_input_masks)
        segment_ids.append(sent_segment_ids)

    assert len(input_ids) == 2

    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_masks = torch.tensor(input_masks).unsqueeze(0)
    sent_segment_ids = torch.tensor(segment_ids).unsqueeze(0)
    return input_ids, input_masks, sent_segment_ids

moe = True
num_labels = 2
reduce_dim = 768


# 修改路径2
model_dir = '/nas2/lishengping/caiyun_projects/two_sentences_classifier/script/train-moe-LCQMC_1215_2/'

bert_config_file = os.path.join(model_dir, 'bert_config.json')
vocab_file = os.path.join(model_dir, 'vocab.txt')
model_path = os.path.join(model_dir, 'pytorch_model.bin.e1.s13499')

bert_config = BertConfig.from_json_file(bert_config_file)
bert_config.reduce_dim = reduce_dim
print(f'bert_config: {bert_config}')
labels_text = ['不相似', '相似']
    
model = TwoSentenceClassifier(bert_config, 
                              num_labels=num_labels, 
                              moe=moe, 
                              os_loss=False)


tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                       do_lower_case=True)


state_dict = torch.load(model_path, map_location='cpu')
remove_prefix_state_dict = {k[7: ]: v for k, v in state_dict.items()}
model.load_state_dict(remove_prefix_state_dict)

for k, v in state_dict.items():
    print(k, v.shape)
    
def predict(text_a:str, text_b:str, label=None):
    input_ids, input_masks, sent_segment_ids = convert_text_to_ids([text_a], [text_b])
    inputs = {'input_ids': input_ids,
             'token_type_ids': sent_segment_ids,
             'attention_mask': input_masks,
             'labels': label}
    with torch.no_grad():
        outputs = model(**inputs)
    loss, logits = outputs
    predicted_class = torch.argmax(logits, dim=-1).item()
    prob = logits.view(-1).tolist()[predicted_class]
    pred_text = labels_text[predicted_class]
    prob = round(prob, 3)
    print(f'预测的标签：{pred_text}, 置信分数: {prob}')
    return pred_text, prob
    

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text_a = data.get("text_a")
    text_b = data.get("text_b")
    
    if not text_a:
        return jsonify({"error": "No text_a provided"}), 400
    
    if not text_b:
        return jsonify({"error": "No text_b provided"}), 400
    
    pred_text, prob = predict(text_a, text_b)
    # result = json.dumps({"label": pred_text, 'prob': prob}, ensure_ascii=False)
    result = json.dumps({"label": pred_text, 'prob': prob}, ensure_ascii=False)
    return result


if __name__ == '__main__':
    port = int(sys.argv[1])
    app.run(debug=True, host='0.0.0.0', port=port)

"""
# 启动命令：
python server.py 5000
# 请求示例：
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text_a": "人和畜生的区别是什么？", "text_b": "人与畜生的区别是什么！"}'
"""
