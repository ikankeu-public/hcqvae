import os
import torch
import random
import numpy as np
from utils.datasets import DatasetLoader

TOPIC_COUNT = 20
DEVICE_IDS = [0]



def run_evaluation(execute):
    for i in range(1,6):
        SEED = 42+(i-1)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        train_dataset_loader = DatasetLoader(
            dataset_name="AgNews",
            batch_size=200,
            seed=SEED
        )

        directory_path = f"./runs/{i}"
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        execute(directory_path,train_dataset_loader)
