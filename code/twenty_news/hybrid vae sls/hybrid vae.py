from torch.nn import functional as F
import pennylane as qml
from torch import nn
import numpy as np
import gensim
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trainer import VAETrainer
from twenty_news.common import DEVICE_IDS, run_evaluation, TOPIC_COUNT


WIRES = 10
dev = qml.device('default.qubit', wires=WIRES)
@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    qml.templates.AmplitudeEmbedding(features=inputs, wires=range(WIRES), normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(WIRES))
    return [qml.expval(op=qml.PauliZ(j)) for j in range(WIRES)]

circuit_weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=WIRES)}

class HybridVAESLS(nn.Module):
    def __init__(self,vocab):
        super(HybridVAESLS, self).__init__()

        vocab_size = len(vocab)
        self.dropout = nn.Dropout(p=.25)
        self.encoder_fc1024 = nn.Linear(vocab_size, 1024)
        self.encoder_fc10_mu = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        self.encoder_fc10_logvar = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        self.gsm_fc = nn.Linear(10, TOPIC_COUNT)

        self.mu_alpha = nn.Parameter(torch.randn(10))
        self.log_var_alpha = nn.Parameter(torch.randn(10))

        self.bn_mu =  nn.BatchNorm1d(
            10, affine=False
        )

        self.bn_log_var =  nn.BatchNorm1d(
            10, affine=False
        )

        self.temperature = nn.Parameter(torch.randn(1).squeeze())

        self.word_embeddings = nn.Linear(300,vocab_size,bias=False)
        self.topic_embeddings = nn.Linear(TOPIC_COUNT,300,bias=False)
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        pretrained_embeddings = torch.tensor([glove_vectors[w] if  w in glove_vectors else np.asarray([1]*300) for i,w in enumerate(vocab) ])
        self.word_embeddings.weight = torch.nn.Parameter(pretrained_embeddings.float())

    def encode(self, batch):
        batch = self.dropout(batch)
        batch = F.tanh(self.encoder_fc1024(batch))

        return self.bn_mu(self.mu_alpha*self.encoder_fc10_mu(batch)),self.bn_log_var(self.log_var_alpha*self.encoder_fc10_logvar(batch))

    def decode(self, batch):
        return F.softmax(self.word_embeddings(self.topic_embeddings(batch)), dim=-1)

    def get_topics(self):
        return F.softmax(self.word_embeddings(self.topic_embeddings.weight.t()), dim=-1)

    def get_topic_embeddings(self):
        return self.topic_embeddings.weight.t()

    def gsm(self,z):
        return F.softmax(self.temperature*self.gsm_fc(z),dim=-1)

    def forward(self, batch):
        mu,log_var = self.encode(batch)
        sigma = log_var.exp().sqrt()
        eta = torch.randn_like(mu) * sigma + mu
        z = self.gsm(eta)
        
        return self.decode(z), mu, sigma.log()

    def get_description(self):
        return f"Model: hybrid vae, Pretrained Model: glove-wiki-gigaword-300, Quantum Circuit: AE->SEL->SEL->SEL-><Z>"
        
        
def execute(director_path,train_dataset_loader):
    trainer = VAETrainer(
        model=HybridVAESLS(train_dataset_loader.get_vocab()),
        dataset_loader=train_dataset_loader,
        epoch=20,
        model_path=f"{director_path}/../hybrid-vae-{director_path.split('/')[-1]}-{{test_acc}}#.pth",
        log_path=f"{director_path}/hybrid-vae.txt",
        device_ids=DEVICE_IDS
    )

    trainer.train()

run_evaluation(execute)
