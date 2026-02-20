import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 770
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
device = torch.device("cpu")

OUT_DIR = "outputs_classification"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Dataset generation (exact spec)
# -----------------------------
def generate_classification_data(seed=SEED, n_samples=500):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2).astype(np.float32)
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    angle = np.arctan2(X[:, 1], X[:, 0])
    y = ((angle + radius * 0.5) % (2 * np.pi) < np.pi).astype(int)

    # flip exactly 10% labels
    flip_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y.astype(np.int64)

X, y = generate_classification_data()

# 70 / 15 / 15 split (stratified)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
val_ratio = 0.15 / (1 - 0.15)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp
)

np.savez(
    os.path.join(OUT_DIR, "classification_data.npz"),
    X_train=X_train, X_val=X_val, X_test=X_test,
    y_train=y_train, y_val=y_val, y_test=y_test
)

# -----------------------------
# Model definition
# -----------------------------
class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = ClassificationNet().to(device)

# -----------------------------
# Training setup
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
EPOCHS = 1500

X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
y_val_t = torch.from_numpy(y_val).float().unsqueeze(1).to(device)
X_test_t = torch.from_numpy(X_test).to(device)
y_test_t = torch.from_numpy(y_test).float().unsqueeze(1).to(device)

train_loss_hist = []
val_loss_hist = []
val_acc_hist = []

# -----------------------------
# Training loop (full batch)
# -----------------------------
for _ in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()
    train_loss_hist.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()
        val_loss_hist.append(val_loss)
        val_acc = ((val_preds >= 0.5).int() == y_val_t.int()).float().mean().item()
        val_acc_hist.append(val_acc)

# -----------------------------
# Test evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    test_probs = model(X_test_t)
    test_labels = (test_probs >= 0.5).int().cpu().numpy().flatten()

test_acc = accuracy_score(y_test, test_labels)
cm = confusion_matrix(y_test, test_labels)

torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_state.pth"))

# -----------------------------
# Plots
# -----------------------------
# Loss history
plt.figure(figsize=(7, 5))
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=150)
plt.close()

# Validation accuracy
plt.figure(figsize=(7, 5))
plt.plot(val_acc_hist, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "val_accuracy.png"), dpi=150)
plt.close()

# Decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
with torch.no_grad():
    probs = model(torch.from_numpy(grid)).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.6)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="Class 0", edgecolor="k")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="Class 1", edgecolor="k")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_boundary.png"), dpi=150)
plt.close()

# Confusion matrix plot
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# -----------------------------
# Report
# -----------------------------
report = (
    f"Final Train Loss: {train_loss_hist[-1]:.4f}\n"
    f"Final Validation Loss: {val_loss_hist[-1]:.4f}\n"
    f"Final Validation Accuracy: {val_acc_hist[-1]:.4f}\n"
    f"Test Accuracy: {test_acc:.4f}\n"
    f"Confusion Matrix:\n{cm}\n"
)

with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
    f.write(report)

print(report)
