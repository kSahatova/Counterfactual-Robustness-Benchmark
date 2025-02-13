import os
from easydict import EasyDict as edict
from typing import Optional, Union, Generator

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torchmetrics import F1Score, MetricCollection


from src.trainers.trainer import AbstractTrainer
from src.utils.visualization import visualize_training_results


# TODO: update the performance metric  

class ClassifierTrainer(AbstractTrainer):
    def __init__(self, opt: edict, model: nn.Module,
                 experiment_path: Optional[str] = None, 
                 restore_training: bool=False):
        super().__init__(self, opt, model, experiment_path, restore_training)

        metric_list = [
            F1Score(num_classes=self.opt.n_classes, task=self.opt.task, average='macro'),
        ]
        self.train_metrics = MetricCollection(metric_list).to(self.device)
        self.val_metrics = self.train_metrics.clone()       
        
        self.logger.info(f'Using model: {self.model.__class__.__name__}')
    

    def training_epoch(self, dataloader: Union[Generator, DataLoader]):
        self.model.train()
        losses = []

        for i, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device).to(torch.long)
            if len(labels.shape) == 2:
                labels = labels.squeeze(1)

            self.opt.optimizer.zero_grad()
            
            outs = self.model(images)   
            loss = self.opt.criterion(outs, labels)
            
            encoded_labels = one_hot(labels, num_classes=self.opt.n_classes)
            self.train_metrics.update(outs, encoded_labels)

            losses.append(loss)
            loss.backward()
            
            self.opt.optimizer.step()

        epoch_stats = {'loss': torch.mean(torch.tensor(losses)), **self.train_metrics.compute()}
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss'],  
               epoch_stats[f'{self.opt.task.title()}F1Score'])
        )
        return epoch_stats
        

    def validation_epoch(self, dataloader: Union[Generator, DataLoader]):
        self.model.eval()
        losses = []
        for images, labels in dataloader:
            
            images = images.to(self.device)            
            if len(images.shape) == 3:
                images = images.unsqueeze(1)

            labels = labels.to(self.device).to(torch.long)
            if len(labels.shape) == 2:
                labels = labels.squeeze(1)

            outs = self.model(images)
            loss = self.opt.criterion(outs, labels)

            encoded_labels = one_hot(labels, num_classes=self.opt.n_classes)
            self.val_metrics.update(outs, encoded_labels)

            losses.append(loss)
            
        epoch_stats = {'loss': torch.mean(torch.tensor(losses)), **self.val_metrics.compute()}
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss'], 
               epoch_stats[f'{self.opt.task.title()}F1Score'])
        )
        return epoch_stats

    def save_state(self):
        model_name = self.model.__class__.__name__ + f'_{self.opt.dataset_name}_{self.opt.task}'
        weights_path = os.path.join(self.ckpt_dir, model_name+f'_{self.current_epoch}_epoch'+'.pth')
        torch.save(self.model.state_dict(), weights_path)
        self.logger.info(f'Saved checkpoint parameters at epoch {self.current_epoch}: {weights_path}')


    def run(self, train_loader, val_loader):
        training_stats = {'loss': [], 'f1score': []}
        val_stats = {'loss': [], 'f1score': []}
        for _ in range(self.current_epoch, self.opt.n_epochs):
            train_epoch_stats = self.training_epoch(train_loader)
            training_stats['loss'].append(train_epoch_stats['loss'])
            training_stats['f1score'].append(train_epoch_stats['BinaryF1Score'])

            with torch.no_grad():
                val_epoch_stats = self.validation_epoch(val_loader)
                val_stats['loss'].append(val_epoch_stats['loss'])
                val_stats['f1score'].append(val_epoch_stats['BinaryF1Score'])

            if self.current_epoch % self.opt.checkpoint_freq == 0 and self.current_epoch != 0:
                self.save_state()                
            
            self.current_epoch += 1
            
        self.save_state()
        if self.opt.visualize_results:
            visualize_training_results(training_stats, val_stats, save_dir=self.vis_dir)


        return training_stats, val_stats