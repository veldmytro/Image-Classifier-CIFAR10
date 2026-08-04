# CIFAR-10 Image Classifier

Image classifier built with PyTorch and trained on the CIFAR-10 dataset.

The project also includes a small experiment exploring overfitting in this particular nn architecture. The results from different training checkpoints can be found in `experiment.txt`.

## Dataset

This project uses the CIFAR-10 dataset.

Download the dataset from:
https://www.cs.toronto.edu/~kriz/cifar.html

Place the dataset inside the `data/` folder.

## Project Structure

- `main.py` - training and evaluation pipeline
- `model.py` - neural network architecture
- `experiment.txt` - results from the overfitting experiment
- `checkpoints/` - saved model weights
- `data/` - CIFAR-10 dataset (not included in the repository)
