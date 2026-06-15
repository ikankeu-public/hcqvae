from torch.nn import functional as F
from torch import nn
import numpy as np
import gensim
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trainer import VAETrainer
from twenty_news.common import DEVICE_IDS, run_evaluation, TOPIC_COUNT


class ClassicalVAELLS(nn.Module):
    def __init__(self,vocab):
        super(ClassicalVAELLS, self).__init__()

        vocab_size = len(vocab)
        self.dropout = nn.Dropout(p=.25)
        self.encoder_fc1024 = nn.Linear(vocab_size, 1024)
        self.encoder_fc10_mu = nn.Linear(1024, 32)
        self.encoder_fc10_logvar = nn.Linear(1024, 32)
        self.gsm_fc = nn.Linear(32, TOPIC_COUNT)

        self.bn_mu =  nn.BatchNorm1d(
            32, affine=False
        )

        self.bn_log_var =  nn.BatchNorm1d(
            32, affine=False
        )

        self.word_embeddings = nn.Linear(300,vocab_size,bias=False)
        self.topic_embeddings = nn.Linear(TOPIC_COUNT,300,bias=False)
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        pretrained_embeddings = torch.tensor([glove_vectors[w] if  w in glove_vectors else np.asarray([1]*300) for i,w in enumerate(vocab) ])
        self.word_embeddings.weight = torch.nn.Parameter(pretrained_embeddings.float())

    def encode(self, batch):
        batch = self.dropout(batch)
        batch = F.tanh(self.encoder_fc1024(batch))
        return self.bn_mu(F.tanh(self.encoder_fc10_mu(batch))),self.bn_log_var(F.tanh(self.encoder_fc10_logvar(batch)))

    def decode(self, batch):
        return F.softmax(self.word_embeddings(self.topic_embeddings(batch)), dim=-1)

    def get_topics(self):
        return F.softmax(self.word_embeddings(self.topic_embeddings.weight.t()), dim=-1)

    def get_topic_embeddings(self):
        return self.topic_embeddings.weight.t()

    def gsm(self,z):
        return F.softmax(self.gsm_fc(z),dim=-1)

    def forward(self, batch):
        mu,log_var = self.encode(batch)
        sigma = log_var.exp().sqrt()
        eta = torch.randn_like(mu) * sigma + mu
        z = self.gsm(eta)
 
        return self.decode(z), mu, sigma.log()

    def get_description(self):
        return f"Model: classical vae (prob), Pretrained Model: glove-wiki-gigaword-300"
        
        
def execute(director_path,train_dataset_loader):
    trainer = VAETrainer(
        model=ClassicalVAELLS(train_dataset_loader.get_vocab()),
        dataset_loader=train_dataset_loader,
        epoch=20,
        model_path=f"{director_path}/../classical-vae-(prob)-{director_path.split('/')[-1]}-{{test_acc}}#.pth",
        log_path=f"{director_path}/classical-vae-(prob).txt",
        device_ids=DEVICE_IDS
    )

    trainer.train()

run_evaluation(execute)
