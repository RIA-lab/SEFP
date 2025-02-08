import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from transformers import EsmTokenizer, EsmModel

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)))
        output = torch.bmm(attn_weights, v)
        return output


class BBAModule(nn.Module):
    def __init__(self):
        super(BBAModule, self).__init__()
        self.bilstm1 = nn.LSTM(input_size=640, hidden_size=320, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=640, hidden_size=320, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(1280, 256)
        self.self_attention = SelfAttention(256)
        self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, embeddings):
        x1, _ = self.bilstm1(embeddings)
        x2, _ = self.bilstm2(x1)
        x2 = torch.cat((x1, x2), dim=-1)

        x2 = self.linear(x2)
        x3 = self.self_attention(x2)
        x3 = torch.cat((x2, x3), dim=-1)
        x3 = self.batchnorm1(x3.permute(0, 2, 1))
        return x3


class CSAttentionModule(nn.Module):
    def __init__(self):
        super(CSAttentionModule, self).__init__()
        self.conv1d = nn.Conv1d(512, 512, 1)
        self.pooling = nn.AdaptiveAvgPool1d(512)

        self.fc1 = nn.Linear(512, 1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x_conv = self.conv1d(x)
        x_conv = x_conv * x

        x_pool = self.pooling(x.permute(0, 2, 1))
        x_pool = torch.nn.functional.relu(self.fc1(x_pool))
        x_pool = self.dropout1(x_pool)
        x_pool = torch.sigmoid(self.fc2(x_pool))
        x_pool = self.dropout2(x_pool)
        x_pool = x_pool.permute(0, 2, 1) * x
        return x_conv + x_pool


class BioBCS(nn.Module):
    def __init__(self, num_labels):
        super(BioBCS, self).__init__()
        self.bba_module = BBAModule()
        self.cs_attention_module = CSAttentionModule()
        self.batchnorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        x = self.bba_module(x)
        x = self.cs_attention_module(x)
        out = self.batchnorm(x).transpose(1, 2)
        out = self.dropout(out)
        logits = self.fc(out[:, 0, :])
        return logits


class Model(nn.Module):
    def __init__(self, esm_checkpoint, num_labels):
        super(Model, self).__init__()
        self.num_labels = num_labels
        self.residue_global_attention = EsmModel.from_pretrained(esm_checkpoint, ignore_mismatched_sizes=True,
                                                                 add_pooling_layer=False)
        self.bio_bcs = BioBCS(num_labels)

    def forward(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = self.residue_global_attention(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeddings = outputs.last_hidden_state

        logits = self.bio_bcs(embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ModelOutput(loss=loss, logits=logits)


class Collator:
    def __init__(self, pretrain_model):
        self.tokenizer = EsmTokenizer.from_pretrained(pretrain_model)

    def __call__(self, batch):
        ids, labels, seqs, structure_features = zip(*batch)
        inputs = self.tokenizer(list(seqs), return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        labels = torch.tensor(list(labels), dtype=torch.long)
        inputs['labels'] = labels
        return inputs