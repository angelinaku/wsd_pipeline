from typing import Dict, List, Any, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from utils.text_processing_utils import Embedder
from utils.utils import load_obj


class WSDLightning(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], conf: DictConfig, tag_to_idx: Dict, embedder: Embedder,
                 num_steps: int = 0):
        super().__init__()

        self.conf = conf
        self.hparams = hparams
        self.tag_to_idx = tag_to_idx
        self.embedder = embedder
        self.num_steps = num_steps

        self.model = load_obj(self.conf.model.model_class)(embeddings_dim=self.conf.data.embedding_shape,
                                                           tag_to_idx=self.tag_to_idx, **self.conf.model.params)
        # if the metric we are using is a class
        if self.conf.training.metric.functional is False:
            self.metric = load_obj(self.conf.training.metric.metric_class)(**self.conf.training.metric.params)

    def forward(self, texts, lengths, *args, **kwargs):
        return self.model(texts, lengths)

    def configure_optimizers(
            self
    ):

        optimizer = load_obj(self.conf.training.optimizer.name)(self.model.parameters(),
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

    def training_step(self, batch, *args, **kwargs):

        sentences, lengths, tags = batch

        embeddings = self.embedder(sentences)

        tag_preds, loss, tag_preds_list = self.model(embeddings, lengths, tags)

        # if the metric we are using is a function
        if self.conf.training.metric.functional:
            # Creating flatten tags list for computing score with sklearn
            tags = tags.flatten().tolist()
            tags_list = [i for i in tags if i != self.tag_to_idx['PAD']]

            metric_score = load_obj(self.conf.training.metric.metric_class)(tags_list, tag_preds_list,
                                                                            **self.conf.training.metric.params)
            metric_score = torch.tensor(metric_score)
        else:
            tags = tags.flatten()
            tags = tags[tags != self.tag_to_idx['PAD']]
            metric_score = self.metric(tag_preds, tags)

        log = {'train_metric': metric_score.item(), 'loss': loss.item()}
        # metric to be logged to a progress bar
        prog_log = {'train_metric': metric_score.item()}

        return {'loss': loss, 'log': log, 'progress_bar': prog_log}

    def validation_step(self, batch, *args, **kwargs):

        sentences, lengths, tags = batch
        embeddings = self.embedder(sentences)

        tag_preds, loss, tag_preds_list = self.model(embeddings, lengths, tags)

        if self.conf.training.metric.functional:
            tags = tags.flatten().tolist()
            tags = [i for i in tags if i != self.tag_to_idx['PAD']]

            metric_score = load_obj(self.conf.training.metric.metric_class)(tags, tag_preds_list,
                                                                            **self.conf.training.metric.params)
            metric_score = torch.tensor(metric_score)
        else:
            tags = tags.flatten()
            tags = tags[tags != self.tag_to_idx['PAD']]
            metric_score = self.metric(tag_preds, tags)

        log = {'valid_loss': loss.item()}

        return {'valid_loss': loss, 'log': log, 'step_metric': metric_score, 'predicted_list': tag_preds_list,
                'predicted_seq': tag_preds, 'true_seq': tags}

    def validation_epoch_end(self, outputs: List[Any]) -> Dict:

        mean_loss = np.stack([x['valid_loss'] for x in outputs]).mean()
        mean_metric = np.stack([x['step_metric'] for x in outputs]).mean()
        # Computing values for a metric

        if self.conf.training.metric.functional:

            true_vals = [x['true_seq'] for x in outputs]
            y_true = [list for sublist in true_vals for list in sublist]

            pred_vals = [x['predicted_list'] for x in outputs]
            y_pred = [list for sublist in pred_vals for list in sublist]

            valid_score = load_obj(self.conf.training.metric.metric_class)(y_true, y_pred,
                                                                           **self.conf.training.metric.params)
            valid_score = torch.tensor(valid_score)
        else:
            y_true = torch.cat([x['true_seq'] for x in outputs])
            y_pred = torch.cat([x['predicted_seq'] for x in outputs])

            valid_score = self.metric(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

        tensorboard_logs = {'valid_score': valid_score, 'valid_score_mean': mean_metric, 'valid_mean_loss': mean_loss}

        return {'validation_loss': mean_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, *args, **kwargs):

        sentences, lengths, tags = batch

        embeddings = self.embedder(sentences)

        tag_preds, loss, tag_preds_list = self.model(embeddings, lengths, tags)

        if self.conf.training.metric.functional:
            tags = tags.flatten().tolist()
            tags = [i for i in tags if i != self.tag_to_idx['PAD']]

            metric_score = load_obj(self.conf.training.metric.metric_class)(tags, tag_preds_list,
                                                                            **self.conf.training.metric.params)
            metric_score = torch.tensor(metric_score)

        else:

            tags = tags.flatten()
            tags = tags[tags != self.tag_to_idx['PAD']]
            metric_score = self.metric(tag_preds, tags)

        log = {'test_loss': loss.item()}

        return {'test_loss': loss, 'log': log, 'step_metric_test': metric_score, 'predicted_list': tag_preds_list,
                'predicted_seq': tag_preds, 'true_seq': tags}

    def test_epoch_end(self, outputs: List[Any]) -> Dict:

        mean_loss = np.stack([x['test_loss'] for x in outputs]).mean()

        # Computing values for a metric
        if self.conf.training.metric.functional:

            true_vals = [x['true_seq'] for x in outputs]
            y_true = [list for sublist in true_vals for list in sublist]

            pred_vals = [x['predicted_list'] for x in outputs]
            y_pred = [list for sublist in pred_vals for list in sublist]

            test_score = load_obj(self.conf.training.metric.metric_class)(y_true, y_pred,
                                                                          **self.conf.training.metric.params)
            test_score = torch.tensor(test_score)

        else:
            y_true = torch.cat([x['true_seq'] for x in outputs])
            y_pred = torch.cat([x['predicted_seq'] for x in outputs])

            test_score = self.metric(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

        # PytorchLightning doesn't like not one-element tensors in the output
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)

        return {'mean_test_loss': mean_loss, 'test_score': test_score, 'predicted': y_true, 'true': y_pred}
