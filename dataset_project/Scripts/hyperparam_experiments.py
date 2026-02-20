import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

SEED = 770
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
device = torch.device("cpu")

OUT_DIR = "outputs_hyperparam"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_classification_data(seed=SEED, n_samples=500):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2).astype(np.float32)
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    angle = np.arctan2(X[:, 1], X[:, 0])
    y = ((angle + radius * 0.5) % (2 * np.pi) < np.pi).astype(int)
    noise_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    return X, y.astype(np.int64)

def split_70_15_15(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    val_ratio = 0.15 / (1 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

class Classifier(nn.Module):
    def __init__(self, act_name: str):
        super().__init__()
        if act_name == "ReLU":
            act = nn.ReLU
        elif act_name == "Sigmoid":
            act = nn.Sigmoid
        elif act_name == "Tanh":
            act = nn.Tanh
        else:
            raise ValueError("Unknown activation")

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            act(),
            nn.Linear(8, 4),
            act(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_eval(act_name, lr, epochs=1500):
    # reset seeds so each run is deterministic
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y = generate_classification_data(SEED, 500)
    X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(X, y)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)
    X_test_t = torch.from_numpy(X_test)

    model = Classifier(act_name).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        probs = model(X_train_t)
        loss = criterion(probs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_t).numpy().flatten()
        val_pred = (val_probs >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_pred)

        test_probs = model(X_test_t).numpy().flatten()
        test_pred = (test_probs >= 0.5).astype(int)
        test_acc = accuracy_score(y_test, test_pred)
        cm = confusion_matrix(y_test, test_pred)

    return val_acc, test_acc, cm

def main():
    activations = ["ReLU", "Sigmoid", "Tanh"]
    lrs = [0.01, 0.1, 0.5]

    results = []
    for act in activations:
        for lr in lrs:
            val_acc, test_acc, cm = train_eval(act, lr)
            results.append((act, lr, val_acc, test_acc, cm))

    # print table
    print("Activation | LR   | Val Acc | Test Acc")
    print("--------------------------------------")
    for act, lr, va, ta, _ in results:
        print(f"{act:<10} | {lr:<4} | {va:<7.4f} | {ta:<8.4f}")

    # save detailed report
    with open(os.path.join(OUT_DIR, "hyperparam_report.txt"), "w") as f:
        f.write("Activation | LR | Val Acc | Test Acc | Confusion Matrix\n\n")
        for act, lr, va, ta, cm in results:
            f.write(f"{act} | {lr} | {va:.4f} | {ta:.4f}\n{cm}\n\n")

if __name__ == "__main__":
    main()
