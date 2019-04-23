import torch
import ipdb
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification


text = "[CLS] The Rock is destined to be the 21st Century."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

num_labels = 5
labels = torch.LongTensor([4])

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

model = BertForSequenceClassification(config, num_labels)
logits = model(tokens_tensor, labels=labels)

softmax = torch.nn.Softmax()
ipdb.set_trace()


