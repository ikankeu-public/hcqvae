import glob
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.nn import functional as F
from tqdm.auto import tqdm

from utils.datasets import DatasetLoader
from utils.utils import DEVICE, compute_metrics


class VAETrainer:

    def __init__(self, model, dataset_loader: DatasetLoader, epoch, model_path, log_path, device_ids=[], continue_at_epoch=None):
        self.model = model
        self.dataset_loader = dataset_loader
        self.epoch = epoch
        self.log_path = log_path
        self.model_path = model_path
        self.device_ids = device_ids
        self.continue_at_epoch = continue_at_epoch

    def train(self):
        self._log("Start Training: "+datetime.now().astimezone().isoformat())
        self._log(self.model.get_description())
        self._log(f"Dataset: {self.dataset_loader.get_dataset_name()}, Batch: {self.dataset_loader.batch_size}, GPU: {self.device_ids}")

        model = self.model
        train_loader = self.dataset_loader.get_loader()

        prev_epoch = 0
        if self.continue_at_epoch is not None and self.model_path:
            model,prev_epoch = self.load_state_dict(self.model_path.replace("{epoch}",str(self.continue_at_epoch)),model)

        model = torch.nn.DataParallel(model,device_ids=self.device_ids)

        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=2e-3)
        criterion = self._criterion
        for epoch in range(prev_epoch+1,self.epoch+1):
            train_loss = self._train(model, train_loader, optimizer, criterion, epoch)
            self._log(f"Epoch {epoch}/{self.epoch}, Train Loss: {train_loss:.4f}")
            if self.model_path and "{test_acc}" in self.model_path:
                existing_files = glob.glob(self.model_path.replace("{test_acc}#","*"))
                assert len(existing_files)<=1, f"Ambiguous models: {existing_files}"

                metrics = self._evaluate(self.model)
                test_accuracy = float('%.6f'%(metrics[0]*metrics[2]))
                self._log(f"Epoch {epoch}/{self.epoch}, CV: {metrics[0]:.4f}, NPMI: {metrics[1]:.4f}, TD: {metrics[2]:.4f}")
                if len(existing_files)==0 or float(existing_files[0].split("-")[-1].split("#")[0]) < test_accuracy:
                    if len(existing_files)>=1:
                        os.remove(existing_files[0])
                    self.save_state_dict(self.model_path.replace("{test_acc}",str(test_accuracy)).replace("#","#"+str(epoch)),model,epoch,self.dataset_loader.get_vocab())
            else:
                if self.model_path:
                    self.save_state_dict(self.model_path.replace("{epoch}",str(epoch)).replace("#","#"+str(epoch)),model,epoch,self.dataset_loader.get_vocab())
                metrics = self._evaluate(self.model)
                self._log(f"Epoch {epoch}/{self.epoch}, CV: {metrics[0]:.4f}, NPMI: {metrics[1]:.4f}, TD: {metrics[2]:.4f}")

        self._log("End Training: "+datetime.now().astimezone().isoformat())
        return model

    def _log(self,msg):
        log_file = open(self.log_path,"a")
        log_file.write(msg+"\n")
        print(msg)
        log_file.close()

    def _standard_normal_kld(self,mu, log_sigma):
        return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)

    def _covariance_penalty(self,topic_embeddings):
        norm_topic = F.normalize(topic_embeddings,dim=-1)
        cosine = (norm_topic @ norm_topic.t()).abs()
        mean = cosine.mean()
        var = ((cosine - mean) ** 2).mean()
        return mean + var


    def _criterion(self,batch, recon_batch, mu,log_sigma):
        recon_loss = -torch.sum(torch.log(torch.clamp(recon_batch,min=1e-16)) * batch,dim=-1)

        kld =  self._standard_normal_kld(mu,log_sigma)
        return (recon_loss + kld).mean() + self._covariance_penalty(self.model.get_topic_embeddings())

    def _train(self, model, train_loader, optimizer, criterion, epoch):
        model.train()
        k = 0
        total_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            batch = batch.to(DEVICE)
            k += batch.shape[0]

            optimizer.zero_grad()
            recon_batch, mu, log_sigma = model(batch)

            loss = criterion(batch, recon_batch, mu, log_sigma)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f'Epoch {epoch}/{self.epoch}')
            loop.set_postfix(loss=loss.item())

        return total_loss / k

    def _evaluate(self, model):
        model.eval()
        with torch.no_grad():
            topics = [self.dataset_loader.get_vocab()[topic.argsort(descending=True)] for topic in model.get_topics().cpu()]
            metrics = compute_metrics(self.dataset_loader.get_corpus(),topics)
        return metrics

    @staticmethod
    def save_state_dict(model_path,model,epoch,vocab):
        torch.save({'epoch': epoch,
                    'vocab': vocab,
                    'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict()},
                   model_path)

    @staticmethod
    def load_state_dict(model_path,model):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        return model,epoch
