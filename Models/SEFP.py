import torch
import torch.nn as nn
from Models.PointNet import PointNetSetAbstraction
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, EsmModel
from typing import Optional
import torch.nn.functional as F


class SEFPOutput(SequenceClassifierOutput):
    def __init__(self,
                 loss=None,
                 logits=None,
                 hidden_states=None,
                 attentions=None,
                 bbc_out=None,
                 bbc_logits=None,
                 point_net_logits=None):
        super().__init__(loss, logits, hidden_states, attentions)
        self.bbc_out = bbc_out
        self.bbc_logits = bbc_logits
        self.point_net_logits = point_net_logits

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


class BioCSAttentionModule(nn.Module):
    def __init__(self):
        super(BioCSAttentionModule, self).__init__()
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



class PointNetModule(nn.Module):
    def __init__(self, num_labels, normal_channel=True):
        super(PointNetModule, self).__init__()
        self.num_labels = num_labels
        in_channel = 640 if normal_channel else 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel + 3,
                                          mlp=[640, 1280, 640], group_all=False)
        self.sa1_linear1 = nn.Linear(1000, 512)
        self.sa1_linear2 = nn.Linear(1024, 512)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=in_channel + 3,
                                          mlp=[640, 1280, 640], group_all=False)
        self.sa2_linear1 = nn.Linear(1000, 128)
        self.sa2_linear2 = nn.Linear(256, 128)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=in_channel + 3,
                                          mlp=[640, 1280, 640], group_all=True)
        self.fc1 = nn.Linear(640, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, num_labels)

    def forward(self, structures,
                embedding_hidden_state9,
                embedding_hidden_state19,
                embedding_hidden_state29, ):
        xyz = structures[:, :3, :]
        B, _, _ = xyz.shape

        l1_xyz, l1_points = self.sa1(xyz, embedding_hidden_state9)
        embedding_hidden_state19 = self.sa1_linear1(embedding_hidden_state19)
        l1_points = torch.cat((l1_points, embedding_hidden_state19), dim=-1)
        l1_points = self.sa1_linear2(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        embedding_hidden_state29 = self.sa2_linear1(embedding_hidden_state29)
        l2_points = torch.cat((l2_points, embedding_hidden_state29), dim=-1)
        l2_points = self.sa2_linear2(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        logits = l3_points.view(B, 640)
        logits = self.drop1(F.relu(self.bn1(self.fc1(logits))))
        logits = self.drop2(F.relu(self.bn2(self.fc2(logits))))
        logits = self.fc3(logits)
        return logits



class SEFP(nn.Module):
    def __init__(self, esm_checkpoint, num_labels):
        super(SEFP, self).__init__()
        self.num_labels = num_labels

        self.esm = EsmModel.from_pretrained(esm_checkpoint, ignore_mismatched_sizes=True, add_pooling_layer=False)
        self.bba = BBAModule()
        self.bca = BioCSAttentionModule()
        self.point_net = PointNetModule(num_labels)

        self.batchnorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(512, num_labels)

        self.fc_fusion = nn.Linear(2 * num_labels, num_labels)
        # self.fc_fusion_dropout = nn.Dropout(0.1)
        # self.fc_fusion2 = nn.Linear(2 * num_labels, num_labels)


    def get_embedding(self, input_ids,
                      attention_mask,
                      position_ids,
                      head_mask,
                      inputs_embeds,
                      encoder_hidden_states,
                      encoder_attention_mask,
                      past_key_values,
                      use_cache,
                      output_attentions,
                      output_hidden_states,
                      return_dict):
        outputs = self.esm(input_ids=input_ids,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask,
                           past_key_values=past_key_values,
                           use_cache=use_cache,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           return_dict=return_dict)
        return outputs

    def forward(self, input_ids: Optional = None,
                attention_mask: Optional = None,
                structures: Optional = None,
                position_ids: Optional = None,
                head_mask: Optional = None,
                inputs_embeds: Optional = None,
                encoder_hidden_states: Optional = None,
                encoder_attention_mask: Optional = None,
                past_key_values: Optional = None,
                use_cache: Optional = None,
                output_attentions: Optional = None,
                output_hidden_states: Optional = True, # set True for PointNet
                return_dict: Optional = None,
                labels: Optional = None):

        with torch.no_grad():
            outputs = self.get_embedding(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          position_ids=position_ids,
                                          head_mask=head_mask,
                                          inputs_embeds=inputs_embeds,
                                          encoder_hidden_states=encoder_hidden_states,
                                          encoder_attention_mask=encoder_attention_mask,
                                          past_key_values=past_key_values,
                                          use_cache=use_cache,
                                          output_attentions=output_attentions,
                                          output_hidden_states=output_hidden_states,
                                          return_dict=return_dict)

        # cls = last_hidden_states[:, 0, :]
        # logits = cls

        embeddings = outputs.last_hidden_state
        bba_out = self.bba(embeddings)
        bca_out = self.bca(bba_out)
        out = self.batchnorm(bca_out).transpose(1, 2)
        out = self.dropout(out)
        bbc_logits = self.fc(out[:, 0, :])




        hidden_states9 = outputs.hidden_states[9].permute(0, 2, 1)
        hidden_states19 = outputs.hidden_states[19].permute(0, 2, 1)
        hidden_states29 = outputs.hidden_states[29].permute(0, 2, 1)
        structures = structures[:, :, :hidden_states29.shape[1]].permute(0, 2, 1)
        point_net_logits = self.point_net(structures, hidden_states9, hidden_states19, hidden_states29)


        logits = torch.cat((bbc_logits, point_net_logits), dim=-1)
        logits = self.fc_fusion(logits)


        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        return SEFPOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            bbc_out=out,
            bbc_logits=bbc_logits,
            point_net_logits=point_net_logits
        )





if __name__ == '__main__':
    from enzyme_dataset import *
    from torch.utils.data import DataLoader


    tokenizer = AutoTokenizer.from_pretrained('../tokenizer-esm2_t30_150M_UR50D')
    dataset_name = '../dataset_seq_struct2'
    dataset_test = EnzymeDataset(f'{dataset_name}/test.json', f'{dataset_name}/label2id.json')
    dataloader = DataLoader(dataset_test, batch_size=4, collate_fn=Collate(tokenizer))
    # model = EsmBBC('../checkpoint-rscb', num_labels=len(dataset_test.label2id))
    checkpoint = '../output/facebook/esm2_t30_150M_UR50D-dataset_seq_struct2pretrain/checkpoint-29304'
    model = SEFP(checkpoint, num_labels=len(dataset_test.label2id))

    with torch.no_grad():
        for batch in dataloader:
            out = model(**batch)
            print(out.logits.shape)
            print(out.logits)
            print(out.loss)
            break

