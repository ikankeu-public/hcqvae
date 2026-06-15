import torch
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.vont_preprocessor import TextProcessor
from sklearn.feature_extraction.text import CountVectorizer

class DatasetLoader:

    def __init__(self,dataset_name,batch_size,seed,test=False):
        self.seed = seed
        self.test = test
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        self.generator = torch.Generator().manual_seed(self.seed)
        self.corpus = None
        self.train_loader = None
        self.vocab = None
        self.init()

    def get_dataset_name(self):
        return self.dataset_name



    def _load_data(self):
        if self.corpus is None:
            from octis.dataset.dataset import Dataset
            if self.dataset_name=="20NewsGroup":
                dataset = Dataset()
                dataset.fetch_dataset("20NewsGroup")
                self.corpus = dataset.get_corpus()
            if self.dataset_name=="20NewsGroup(vont)":
                dataset = Dataset()
                dataset.fetch_dataset("20NewsGroup")
                self.corpus = dataset.get_corpus()

        if self.dataset_name=="AgNews":
            from datasets import load_dataset
            df = load_dataset('xwjzds/ag_news')
            self.corpus = [text.split(' ') for text in df['test' if self.test else 'train']['text']]

        return self.corpus

    def get_corpus(self):
        return self.corpus

    def get_vocab(self):
        return self.vocab

    def get_loader(self):
        return self.train_loader

    def init(self):
        if self.train_loader is not None:
            return self.train_loader
        self._load_data()

        if self.dataset_name=="20NewsGroup":
            vectorizer = CountVectorizer(analyzer="word")
            texts = [' '.join(words) for words in self.corpus]
            vectorizer.fit_transform(texts)
            self.vocab = vectorizer.get_feature_names_out()
            self.train_loader = DataLoader(DatasetRaw(texts), batch_size=self.batch_size,collate_fn=CollateRaw(vectorizer), shuffle=True, generator=self.generator)

        if self.dataset_name=="20NewsGroup(vont)":
            text_processor = TextProcessor([' '.join(words) for words in self.corpus])
            text_processor.process()
            self.corpus = text_processor.lemmas
            self.vocab = np.array([w for _,w in sorted(text_processor.index_to_word.items(),key=lambda x: x[0])])
            self.train_loader = DataLoader(DatasetVont(text_processor), batch_size=self.batch_size,collate_fn=CollateVont(), shuffle=True, generator=self.generator)

        if self.dataset_name=="AgNews":
            text_processor = TextProcessor([' '.join(words) for words in self.corpus])
            text_processor.process()
            self.corpus = text_processor.lemmas
            self.vocab = np.array([w for _,w in sorted(text_processor.index_to_word.items(),key=lambda x: x[0])])
            self.train_loader = DataLoader(DatasetVont(text_processor), batch_size=self.batch_size,collate_fn=CollateVont(), shuffle=True, generator=self.generator)

        print("Dataset:",self.get_dataset_name(),"Corpus:",len(self.corpus),"Vocab:",len(self.vocab))

        return self.train_loader


class DatasetVont(Dataset):
    def __init__(self, text_processor):
        self.text_processor = text_processor

    def __len__(self):
        return len(self.text_processor.bow)

    def __getitem__(self, index):
        return self.text_processor.bow[index]

class CollateVont:
    def __call__(self,inputs):
        return tensor(inputs).float()

class DatasetRaw(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]

class CollateRaw:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def __call__(self,inputs):
        return tensor(list(self.vectorizer.transform(inputs).toarray())).float()
