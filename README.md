# Hybrid Classical-Quantum VAE for Neural Topic Modeling

This repository contains the implementation used for the paper [**Hybrid Classical-Quantum Variational Autoencoder for Neural Topic Modeling**](https://arxiv.org/abs/2606.13852). The project studies how parameterized quantum circuits can be embedded inside a variational autoencoder (VAE) inference network.

The models train on bag-of-words document representations and learn 20 latent topics. The hybrid VAE uses PennyLane quantum circuits in the encoder to parameterize the mean and variance of a Gaussian Softmax posterior, while the decoder remains classical and reconstructs documents through topic and word embedding matrices. Fully classical VAE variants are included as baselines.

## Repository Structure

- `code/` contains the neural topic modeling implementation.
- `code/ag_news/` contains experiments for the AgNews benchmark.
- `code/twenty_news/` contains experiments for the 20News benchmark.
- `code/utils/` contains shared dataset loading, preprocessing, evaluation, and training utilities.
- `code/*_Result_Analysis.ipynb` notebooks are used to post-process logs and plot results.

Each dataset folder contains four executable experiment folders:

- `classical vae sls/`: classical VAE with a 10-dimensional small latent space.
- `classical vae lls/`: classical VAE with a 32-dimensional large latent space.
- `hybrid vae sls/`: hybrid VAE with a 10-qubit circuit and Pauli-Z expectation measurements.
- `hybrid vae lls/`: hybrid VAE with a 10-qubit circuit and probability measurements over five qubits.
- `models/` contains trained models.

## Dependencies

The Python dependencies are listed in `code/requirements.txt`. Install them from the repository root with:

```bash
pip install -r code/requirements.txt
```

The experiments use `torch`, `pennylane`, `octis`, `datasets`, `gensim`, `numpy`, and `nltk`. A CUDA-capable GPU is recommended. The quantum experiments are simulated with PennyLane's noiseless `default.qubit` backend, but the circuit design is compatible with parameter-shift or finite-difference gradients for NISQ-oriented execution.

## Datasets

The code loads benchmark datasets automatically:

- AgNews is loaded through the Hugging Face `datasets` package.
- 20News is loaded through `octis`.

Preprocessing follows the [vONTSS](https://github.com/xuweijieshuai/vONTSS) preprocessing pipeline used by prior state-of-the-art neural topic modeling work. The shared implementation is in `code/utils/vont_preprocessor.py`. The first run may need internet access to download datasets and GloVe embeddings.

## Running Experiments

Run each experiment from inside its own folder so relative `runs/` paths resolve correctly. For example:

```bash
cd "code/ag_news/hybrid vae sls"
python "hybrid vae.py"
```

Other examples:

```bash
cd "code/ag_news/classical vae lls"
python "classical vae.py"

cd "code/twenty_news/hybrid vae lls"
python "hybrid vae.py"
```

Each script performs five runs with seeds 42 through 46, trains for 20 epochs, uses a batch size of 200, and evaluates after every epoch.

## Outputs and Logs

Experiment outputs are written under each experiment's `runs/` directory. Logs follow this format:

```text
Start Training: [date time]
Model: [model description], Pretrained Model: glove-wiki-gigaword-300
Dataset: [dataset name], Batch: [size], GPU: [device ids]
Epoch [current]/[total], Train Loss: [training loss]
Epoch [current]/[total], CV: [c_v coherence], NPMI: [npmi], TD: [topic diversity]
End Training: [date time]
```

When model checkpoints are saved, they use PyTorch `.pth` files containing the epoch, vocabulary, and model parameters. The stored vocabulary is important for reproducibility because preprocessing can produce vocabulary indices in a non-deterministic order.

## Evaluation

The implementation evaluates topic quality with standard topic modeling metrics:

- `C_v` coherence over the top 10 topic words.
- NPMI over the top 10 topic words.
- Topic Diversity over the top 25 topic words.

The paper reports that the hybrid VAE with the small latent space reaches strong AgNews performance, including approximately `0.71` `C_v`, `0.20` NPMI, and high topic diversity. On 20News, the models obtain coherence scores comparable to or slightly above prior state-of-the-art results, with lower topic diversity.

## Citation

If you use this work, please cite:

```bibtex
@misc{kankeu2026hybridclassicalquantumvariationalautoencoder,
      title={Hybrid Classical-Quantum Variational Autoencoder for Neural Topic Modeling}, 
      author={Ivan Kankeu},
      year={2026},
      eprint={2606.13852},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.13852}, 
}
```