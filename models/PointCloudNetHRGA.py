import torch
import torch.nn as nn
from models.PointNet import PointNetSetAbstraction
from transformers.modeling_outputs import ModelOutput
from transformers import EsmTokenizer, EsmModel
import torch.nn.functional as F


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


class PointNetHRGA(nn.Module):
    def __init__(self, num_labels):
        super(PointNetHRGA, self).__init__()
        self.point_net = PointNetModule(num_labels)

    def forward(self, hidden_states, structures):
        hidden_states9 = hidden_states[9].permute(0, 2, 1)
        hidden_states19 = hidden_states[19].permute(0, 2, 1)
        hidden_states29 = hidden_states[29].permute(0, 2, 1)
        structures = structures[:, :, :hidden_states29.shape[1]]
        logits = self.point_net(structures, hidden_states9, hidden_states19, hidden_states29)

        return logits


class Model(nn.Module):
    def __init__(self, esm_checkpoint, num_labels):
        super(Model, self).__init__()
        self.num_labels = num_labels
        self.residue_global_attention = EsmModel.from_pretrained(esm_checkpoint, ignore_mismatched_sizes=True, add_pooling_layer=False)
        self.point_net_hrga = PointNetHRGA(num_labels)

    def forward(self, input_ids, attention_mask, structures, labels):
        with torch.no_grad():
            outputs = self.residue_global_attention(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        structures = structures[:, :, :hidden_states[-1].shape[1]]
        logits = self.point_net_hrga(hidden_states, structures)

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

        structure_features = list(structure_features)
        for idx, item in enumerate(structure_features):
            if item is None:
                structure_features[idx] = [[float(0)] * 4] * 1000

        inputs['structures'] = torch.tensor(structure_features)
        return inputs

