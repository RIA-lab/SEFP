import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers import EsmTokenizer, EsmModel
import torch.nn.functional as F
from models.BioBCS import BioBCS
from models.PointCloudNetHRGA import PointNetHRGA


class Model(nn.Module):
    def __init__(self, esm_checkpoint, num_labels):
        super(Model, self).__init__()
        self.num_labels = num_labels
        self.residue_global_attention = EsmModel.from_pretrained(esm_checkpoint, ignore_mismatched_sizes=True, add_pooling_layer=False)
        self.point_net_hrga = PointNetHRGA(num_labels)
        self.bio_bcs = BioBCS(num_labels)
        self.mlp = nn.Linear(2 * num_labels, num_labels)

    def forward(self, input_ids, attention_mask, structures, labels):
        with torch.no_grad():
            outputs = self.residue_global_attention(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = outputs.last_hidden_state

        structures = structures[:, :, :hidden_states[-1].shape[1]]
        point_net_logits = self.point_net_hrga(hidden_states, structures)
        bio_bcs_logits = self.bio_bcs(last_hidden_state)
        logits = self.mlp(torch.cat([point_net_logits, bio_bcs_logits], dim=-1))

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
        # for no pdb data set coordinates to zero
        for idx, item in enumerate(structure_features):
            if item is None:
                structure_features[idx] = [[float(0)] * 4] * 1000

        inputs['structures'] = torch.tensor(structure_features)
        return inputs