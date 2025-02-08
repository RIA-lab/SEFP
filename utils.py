from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve
from safetensors import safe_open
import numpy as np
import json
import os


def metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


# freeze the model parameters
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def load_safetonsors_model(model, checkpoint_path):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)


def load_model_part(model, checkpoint_path, part):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            if part == key.split('.')[0]:
                state_dict[key] = f.get_tensor(key)
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)


def load_best_model(model, checkpoint_dir, parts=None, last_is_best=False):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
    if last_is_best:
        best_checkpoint = checkpoints[-1]
        best_checkpoint = os.path.join(checkpoint_dir, best_checkpoint, 'model.safetensors')
    else:
        last_checkpoint = checkpoints[-1]
        trainer_state = load_json(f'{checkpoint_dir}/{last_checkpoint}/trainer_state.json')
        best_checkpoint = trainer_state['best_model_checkpoint']
        best_checkpoint = os.path.join(best_checkpoint, 'model.safetensors')

    print(f'best_checkpoint: {best_checkpoint}')
    if parts is not None:
        for part in parts:
            load_model_part(model, best_checkpoint, part)
    else:
        load_safetonsors_model(model, best_checkpoint)

