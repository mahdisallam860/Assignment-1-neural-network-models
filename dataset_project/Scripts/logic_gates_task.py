import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# -----------------------------
# Reproducibility (seed 770)
# -----------------------------
SEED = 770
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
device = torch.device("cpu")

OUT_DIR = "outputs_logic_gates"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Data: truth table (fixed)
# -----------------------------
def generate_logic_gate_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    gates = {
        "AND":  np.array([0, 0, 0, 1], dtype=np.int64),
        "OR":   np.array([0, 1, 1, 1], dtype=np.int64),
        "NAND": np.array([1, 1, 1, 0], dtype=np.int64),
        "XOR":  np.array([0, 1, 1, 0], dtype=np.int64),
    }
    return X, gates

X_np, gates = generate_logic_gate_data()
X_t = torch.from_numpy(X_np).to(device)

# -----------------------------
# Models
# -----------------------------
class Perceptron(nn.Module):
    # single-layer perceptron: 2 -> 1 with Sigmoid
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class XORNet(nn.Module):
    # 2 -> 2(sigmoid) -> 1(sigmoid)
    def __init__(self, hidden_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Helpers
# -----------------------------
def predict_labels(model, X):
    model.eval()
    with torch.no_grad():
        probs = model(X).cpu().numpy().flatten()
    return (probs >= 0.5).astype(int), probs

def train_until_100(model, y_np, lr=0.1, max_epochs=20000):
    y_t = torch.from_numpy(y_np).float().unsqueeze(1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        probs = model(X_t)
        loss = criterion(probs, y_t)
        loss.backward()
        optimizer.step()

        pred, _ = predict_labels(model, X_t)
        acc = accuracy_score(y_np, pred)
        if acc == 1.0:
            return epoch, loss.item(), acc
    return None, loss.item(), acc

def train_fixed_epochs(model, y_np, lr=0.1, epochs=2000):
    y_t = torch.from_numpy(y_np).float().unsqueeze(1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        probs = model(X_t)
        loss = criterion(probs, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    pred, _ = predict_labels(model, X_t)
    acc = accuracy_score(y_np, pred)
    return losses, acc

def save_weights_heatmaps(model, prefix):
    # Visualize weights for each Linear layer as a heatmap
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()
            b = m.bias.detach().cpu().numpy()

            plt.figure(figsize=(5, 2.8))
            plt.imshow(W, cmap="coolwarm", aspect="auto")
            plt.colorbar()
            plt.title(f"{prefix} - Linear{layer_idx} weight matrix")
            plt.xlabel("Input feature")
            plt.ylabel("Neuron")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{prefix}_linear{layer_idx}_weights.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(5, 2.2))
            plt.bar(np.arange(len(b)), b, color="C0")
            plt.title(f"{prefix} - Linear{layer_idx} bias vector")
            plt.xlabel("Neuron")
            plt.ylabel("Bias")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{prefix}_linear{layer_idx}_bias.png"), dpi=150)
            plt.close()

            layer_idx += 1

def save_xor_search_log(search_rows):
    path = os.path.join(OUT_DIR, "xor_minimal_search.txt")
    with open(path, "w") as f:
        f.write("Minimal XOR architecture search (all use Sigmoid hidden + Sigmoid output, BCE loss)\n")
        f.write("Format: hidden_size | lr | epochs | final_acc | final_loss\n\n")
        for row in search_rows:
            f.write(f"{row['hidden_size']:>10} | {row['lr']:<3} | {row['epochs']:<5} | {row['acc']:<8.4f} | {row['loss']:<.6f}\n")
        f.write("\nChosen minimal architecture is the smallest hidden_size achieving 100% accuracy.\n")
    return path

# -----------------------------
# Run experiments
# -----------------------------
report_lines = []
report_lines.append(f"Seed used: {SEED}\n")

# 0.7.1 Linearly separable gates: AND, OR, NAND (train until 100%)
for gate_name in ["AND", "OR", "NAND"]:
    y = gates[gate_name]
    model = Perceptron().to(device)
    epoch_reached, final_loss, final_acc = train_until_100(model, y, lr=0.1, max_epochs=20000)

    pred, probs = predict_labels(model, X_t)
    acc = accuracy_score(y, pred)

    save_weights_heatmaps(model, f"{gate_name}_perceptron")

    report_lines.append(f"{gate_name} (Perceptron 2->1 Sigmoid, BCE, SGD lr=0.1):")
    report_lines.append(f"  epochs_to_100%: {epoch_reached}")
    report_lines.append(f"  final_loss: {final_loss:.6f}")
    report_lines.append(f"  final_accuracy: {acc:.4f}")
    report_lines.append(f"  predictions: {pred.tolist()}")
    report_lines.append("")

# 0.7.2 XOR: 2->2(sigmoid)->1(sigmoid), BCE, train 2000 epochs (fixed)
xor_y = gates["XOR"]
xor_model = XORNet(hidden_size=2).to(device)
xor_losses, xor_acc = train_fixed_epochs(xor_model, xor_y, lr=0.1, epochs=2000)
xor_pred, xor_probs = predict_labels(xor_model, X_t)

save_weights_heatmaps(xor_model, "XOR_2hidden")
plt.figure(figsize=(7, 4))
plt.plot(xor_losses)
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("XOR Training Loss (2 hidden neurons, 2000 epochs)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xor_loss_curve.png"), dpi=150)
plt.close()

report_lines.append("XOR (2->2 Sigmoid->1 Sigmoid, BCE, SGD lr=0.1, 2000 epochs):")
report_lines.append(f"  final_accuracy: {xor_acc:.4f}")
report_lines.append(f"  predictions: {xor_pred.tolist()}")
report_lines.append("")

# 0.7.3 Minimal architecture challenge (smallest network that solves XOR at 100%)
# Search hidden_size = 1..5 with a few learning rates; keep architecture Sigmoid/Sigmoid as required.
search_rows = []
best = None

for hidden_size in [1, 2, 3, 4, 5]:
    for lr in [0.1, 0.5, 1.0]:
        # reset seed so each trial is deterministic and comparable
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        trial = XORNet(hidden_size=hidden_size).to(device)
        y_t = torch.from_numpy(xor_y).float().unsqueeze(1).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(trial.parameters(), lr=lr)

        final_loss = None
        for _ in range(2000):
            trial.train()
            optimizer.zero_grad()
            probs = trial(X_t)
            loss = criterion(probs, y_t)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        pred, _ = predict_labels(trial, X_t)
        acc = accuracy_score(xor_y, pred)

        row = {"hidden_size": hidden_size, "lr": lr, "epochs": 2000, "acc": acc, "loss": final_loss}
        search_rows.append(row)

        if acc == 1.0 and best is None:
            best = {"hidden_size": hidden_size, "lr": lr}
            best_model = trial

    if best is not None:
        break

search_log_path = save_xor_search_log(search_rows)

if best is not None:
    save_weights_heatmaps