import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import random

SEED = 770
OUT_DIR = "outputs_regression"
os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cpu")

def generate_regression_data(seed, n_samples=500):
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, (n_samples, 2)).astype(np.float32)
    y_true = 2 * X[:,0]**2 - 1.5 * X[:,0] * X[:,1] + 0.8 * X[:,1]**2
    y = y_true + np.random.normal(0, 2, n_samples)
    return X, y.astype(np.float32), y_true.astype(np.float32)

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, X_test, y_test, lr=0.01, epochs=1000):
    model.to(device)
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).unsqueeze(1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()
            test_losses.append(test_loss)

    return train_losses, test_losses, model

def plot_loss(train_losses, test_losses, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='train MSE')
    plt.plot(test_losses, label='test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Test Loss (MSE)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_3d_predictions(X_test, y_test, model, y_true_func, out_path):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy().flatten()
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test[:,0], X_test[:,1], y_test, c='blue', label='actual (noisy)', alpha=0.6)
    ax.scatter(X_test[:,0], X_test[:,1], preds, c='red', label='predicted', alpha=0.6)
    x1 = np.linspace(X_test[:,0].min(), X_test[:,0].max(), 60)
    x2 = np.linspace(X_test[:,1].min(), X_test[:,1].max(), 60)
    X1g, X2g = np.meshgrid(x1, x2)
    Z = y_true_func(X1g, X2g)
    ax.plot_surface(X1g, X2g, Z, cmap='viridis', alpha=0.3)
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    X, y, y_true = generate_regression_data(SEED, n_samples=500)
    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        X, y, y_true, test_size=0.2, random_state=SEED)

    model = RegressionNet()
    train_losses, test_losses, trained_model = train_model(model, X_train, y_train, X_test, y_test,
                                                           lr=0.01, epochs=1000)

    trained_model.eval()
    with torch.no_grad():
        train_pred = trained_model(torch.from_numpy(X_train).to(device)).cpu().numpy().flatten()
        test_pred = trained_model(torch.from_numpy(X_test).to(device)).cpu().numpy().flatten()
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    loss_path = os.path.join(OUT_DIR, "regression_loss.png")
    plot_loss(train_losses, test_losses, loss_path)
    pred3d_path = os.path.join(OUT_DIR, "regression_pred_vs_actual_3d.png")
    y_true_func = lambda x1, x2: 2 * x1**2 - 1.5 * x1 * x2 + 0.8 * x2**2
    plot_3d_predictions(X_test, y_test, trained_model, y_true_func, pred3d_path)

    print(f"Seed used: {SEED}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Saved loss curve: {loss_path}")
    print(f"Saved 3D predictions plot: {pred3d_path}")

if __name__ == "__main__":
    main()
