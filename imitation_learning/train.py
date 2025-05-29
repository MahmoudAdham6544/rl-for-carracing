from collections import defaultdict
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch

import sys

sys.path.append("./")

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if os.path.getsize(data_file) == 0:
        raise ValueError(f"Data file is empty: {data_file}")

    try:
        with gzip.open(data_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_file}: {e}")

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )

    # print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=0):

    # TODO: preprocess your data here.
    # Convert RGB to grayscale and normalize (already shape: (N, C, H, W))
    X_train = np.array([utils.rgb2gray(x) for x in X_train]) / 255.0
    X_valid = np.array([utils.rgb2gray(x) for x in X_valid]) / 255.0

    def add_history(X, y, history_length):
        """
        Add history to the input data
        """
        X_stack = []
        y_stack = []
        for i in range(history_length - 1, len(X)):
            frames = X[i - history_length + 1 : i + 1]
            X_stack.append(np.stack(frames, axis=0))
            y_stack.append(y[i])
        return np.array(X_stack), np.array(y_stack)

    y_train = np.array([utils.action_to_id(a) for a in y_train])
    y_valid = np.array([utils.action_to_id(a) for a in y_valid])

    # stack history
    if history_length > 1:
        X_train, y_train = add_history(X_train, y_train, history_length)
        X_valid, y_valid = add_history(X_valid, y_valid, history_length)
    else:
        X_train = X_train[:, np.newaxis, :, :]
        X_valid = X_valid[:, np.newaxis, :, :]

    from collections import defaultdict

    class_indices = defaultdict(list)
    for idx, label in enumerate(y_train):
        class_indices[label].append(idx)

    # Step 2: Define oversampling factors per class
    oversample_factors = {
        0: 1,  # Keep dominant class as is
        1: 2,  # Oversample class 1 moderately
        2: 2,  # Oversample class 2 more
        3: 5,  # Oversample class 3 the most
    }

    # Step 3: Collect balanced indices
    balanced_indices = []

    for label, indices in class_indices.items():
        factor = oversample_factors.get(label, 1)
        num_samples = int(len(indices) * factor)

        # Sample with replacement if needed
        sampled = np.random.choice(indices, num_samples, replace=True)
        balanced_indices.extend(sampled)

    # Step 4: Shuffle
    np.random.shuffle(balanced_indices)

    # Step 5: Apply new dataset
    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    return X_train, y_train, X_valid, y_valid


def train_model(
    X_train,
    y_train,
    X_valid,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent()
    tensorboard_eval = Evaluation(
        tensorboard_dir,
        f"Imitation Learning {history_length}",
        stats=["train/loss", "train/accuracy", "val/loss", "val/accuracy"],
    )

    def sample_minibatch(X, y, batch_size):
        """
        This method samples a minibatch of data
        """
        action_to_indices = defaultdict(list)
        for i, label in enumerate(y):
            action_to_indices[label].append(i)
        classes = list(action_to_indices.keys())
        samples_per_class = batch_size // len(classes)
        indices = []
        for c in classes:
            indices.extend(
                np.random.choice(action_to_indices[c], samples_per_class, replace=False)
            )
        np.random.shuffle(indices)
        return X[indices], y[indices]

    # training loop
    for i in range(1, n_minibatches + 1):
        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)
        loss = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            X_tensor = torch.as_tensor(X_batch).clone().detach().to(agent.device)
            y_tensor = torch.as_tensor(y_batch, dtype=torch.long).to(agent.device)

            with torch.no_grad():
                logits = agent.net(X_tensor)
                train_loss = agent.criterion(logits, y_tensor).item()
                train_pred = torch.argmax(logits, dim=1)
                train_accuracy = (train_pred == y_tensor).float().mean().item()

            # --Validation--
            X_val_batch, y_val_batch = sample_minibatch(X_valid, y_valid, batch_size)
            X_val_tensor = (
                torch.as_tensor(X_val_batch).clone().detach().to(agent.device)
            )
            y_val_tensor = torch.as_tensor(y_val_batch, dtype=torch.long).to(
                agent.device
            )
            with torch.no_grad():
                val_logits = agent.net(X_val_tensor)
                val_loss = agent.criterion(val_logits, y_val_tensor).item()
                val_pred = torch.argmax(val_logits, dim=1)
                val_accuracy = (val_pred == y_val_tensor).float().mean().item()

            print(
                f"Iteration {i}/{n_minibatches} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Validation Loss: {val_loss:.4f} | "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )
            tensorboard_eval.write_episode_data(
                i,
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                },
            )

    # TODO save your agent
    model_dir = os.path.join(model_dir, "agent.pt")
    agent.save(model_dir)
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    history_length = 0
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length
    )
    print(f"X_train shape: {X_train.shape}")

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=3000, batch_size=64, lr=1e-4)
