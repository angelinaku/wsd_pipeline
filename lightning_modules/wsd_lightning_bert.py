from typing import Dict

from utils.utils import load_obj
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig


class TextClassificationLightning(pl.LightningModule):

    def __init__(self, hparams: Dict[str, float], conf: DictConfig, num_steps: int = 0):
        super().__init__()

        self.conf = conf
        self.hparams = hparams
        self.num_steps = num_steps
        self.model = load_obj(self.conf.model.model_class)(pretrained_model_name=self.conf.model.model_name,
                                                           num_classes=self.conf.data.num_classes)

        if self.conf.data.num_classes == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # if the metric we are using is a class
        if self.conf.training.metric.functional is False:
            self.metric = load_obj(self.conf.training.metric.metric_class)(**self.conf.training.metric.params)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, *args, **kwargs):

        return self.model(x)

    def configure_optimizers(self):

        optimizer = load_obj(self.conf.training.optimizer.name)(self.wsd_model.parameters(),
                                                                **self.conf.training.optimizer.params)

        if 'transformers.get_linear_schedule_with_warmup' not in self.conf.training.scheduler.name:
            scheduler = load_obj(self.conf.train_setup.scheduler.name)(
                optimizer, **self.conf.train_setup.scheduler.params
            )
            scheduler_dict = {
                'scheduler': scheduler,
                'interval': self.conf.train_setup.scheduler.step,
                'monitor': self.conf.train_setup.scheduler.monitor,
                'name': 'scheduler',
            }
        else:

            num_train_steps = self.num_steps * (self.conf.trainer.min_epochs + 7)
            num_warm = round(num_train_steps * 0.1)
            scheduler = load_obj(self.conf.train_setup.scheduler.name)(
                optimizer, num_training_steps=num_train_steps, num_warmup_steps=num_warm
            )
            scheduler_dict = {'scheduler': scheduler, 'name': 'scheduler'}

        return [optimizer], [scheduler_dict]

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, target_mask, labs = batch

        logits = self.model(input_ids, attention_mask=attention_mask,
                            target_mask=target_mask)

        if self.conf.data.num_classes == 2:
            labs = labs.unsqueeze(1)

        loss = self.criterion(logits, labs.float())

        if self.conf.data.num_classes != 2:
            pred_classes = torch.argmax(self.softmax(logits), dim=1)
        else:
            pred_classes = torch.round(torch.sigmoid(logits)).squeeze(1)

        labs = labs.flatten()
        f1_score = self.metric(pred_classes, labs)

        log = {'f1_score': f1_score, 'train_loss': loss.item()}
        # logging f1 score to progress bar
        self.log('f1_score', f1_score.item(), prog_bar=True, logger=True)

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, target_mask, labs = batch

        logits = self.model(input_ids, attention_mask=attention_mask,
                            target_mask=target_mask)
        if self.conf.data.num_classes == 2:
            labs = labs.unsqueeze(1)

        loss = self.criterion(logits, labs.float())

        if self.conf.data.num_classes != 2:
            pred_classes = torch.argmax(self.softmax(logits), dim=1)
        else:
            pred_classes = torch.round(torch.sigmoid(logits)).squeeze(1)

        labs = labs.flatten()

        f1_score = self.metric(pred_classes, labs)

        log = {'valid_loss': loss.item()}

        return {'valid_loss': loss, 'log': log, 'step_metric': f1_score,
                'predicted': pred_classes, 'true': labs}

    def validation_epoch_end(self, outputs):

        mean_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()

        mean_metric = torch.stack([x['step_metric'] for x in outputs]).mean()

        y_true = torch.cat([x['true'] for x in outputs])
        y_pred = torch.cat([x['predicted'] for x in outputs])

        f1_score = self.metric(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

        self.log('val_f1_mean', mean_metric, prog_bar=True, logger=True)
        self.log('val_f1', f1_score, prog_bar=True, logger=True)
        self.log('valid_mean_loss', mean_loss, prog_bar=True)

        return {'validation_loss': mean_loss, 'val_f1': f1_score,
                'val_f1_mean': mean_metric}

    def test_step(self, batch, batch_idx):

        input_ids, attention_mask, target_mask, labs = batch

        logits = self.model(input_ids, attention_mask=attention_mask,
                            target_mask=target_mask)
        if self.conf.data.num_classes == 2:
            labs = labs.unsqueeze(1)

        loss = self.criterion(logits, labs.float())

        if self.conf.data.num_classes != 2:
            pred_classes = torch.argmax(self.softmax(logits), dim=1)
        else:
            pred_classes = torch.round(torch.sigmoid(logits)).squeeze(1)
        labs = labs.flatten()

        f1_score = self.metric(pred_classes, labs)

        log = {'test_loss': loss.item()}

        return {'test_loss': loss, 'log': log, 'step_metric_test': f1_score,
                'predicted': pred_classes, 'true': labs}

    def test_epoch_end(self, outputs):

        mean_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        mean_metric = torch.stack([x['step_metric_test'] for x in outputs]).mean()

        y_true = torch.cat([x['true'] for x in outputs])
        y_pred = torch.cat([x['predicted'] for x in outputs])

        f1_score = self.metric(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

        self.log('test_f1_mean', mean_metric, prog_bar=True)
        self.log('test_f1', f1_score, prog_bar=True)
        self.log('test_mean_loss', mean_loss, prog_bar=True)

        return {'test_loss': mean_loss, 'test_f1': f1_score}
