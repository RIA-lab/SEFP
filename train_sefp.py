import wandb
from transformers import Trainer, TrainingArguments
from models.SEFP import Model, Collator
from utils import *
from dataset_enzyme import DatasetEnzyme
from torch.optim import AdamW
import os
# os.environ['WANDB_MODE'] = 'offline'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# wandb sync directory


dataset_path = 'data/RSCB_dataset'
folds = os.listdir(dataset_path)


if __name__ == '__main__':
    for fold in folds:
        print(fold)
        dataset_train = DatasetEnzyme(f'{dataset_path}/{fold}/train.json', f'{dataset_path}/{fold}/label2id.json')
        dataset_test = DatasetEnzyme(f'{dataset_path}/{fold}/test.json', f'{dataset_path}/{fold}/label2id.json')
        print(len(dataset_train))
        print(len(dataset_test))

        model = Model('esm2', num_labels=len(dataset_train.label2id))
        freeze_model(model.residue_global_attention)

        lr = 1e-4
        num_epochs = 32
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        wandb.init(project='SEFP')
        wandb.run.name = f'SEFP-{dataset_path.split("/")[-1]}-{fold}'

        args = TrainingArguments(
            output_dir=f'output/{wandb.run.name}',
            logging_dir=f'output/{wandb.run.name}/log',
            logging_strategy='epoch',
            learning_rate=lr,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=num_epochs,
            weight_decay=1e-4,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            run_name=wandb.run.name,
            overwrite_output_dir=True,
            save_total_limit=3,
            remove_unused_columns=False,
            report_to=["wandb"],
            fp16=True,
            # metric_for_best_model='f1',
            # greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            optimizers=(optimizer, None),
            args=args,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            data_collator=Collator('esm2'),
            compute_metrics=metrics,
        )

        trainer.train(resume_from_checkpoint=False)
        wandb.finish()