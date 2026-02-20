#!/usr/bin/env python3
# generate_datasets.py
# Windows 10 ready script that uses seed = 770 and saves + displays all required figures.
# Requirements: numpy, matplotlib, scikit-learn (only for optional checks). Install in your venv.

import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SEED = 770
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_regression_data(seed=SEED, n_samples=500):
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, (n_samples, 2))
    y_true = 2 * X[:, 0] ** 2 - 1.5 * X[:, 0] * X[:, 1] + 0.8 * X[:, 1] ** 2
    y = y_true + np.random.normal(0, 2, n_samples)
    return X, y, y_true


def plot_regression_3d(X, y, y_true_func, save_path=None, elev=30, azim=45):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, c="red", marker="o", alpha=0.6, label="noisy samples")
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 80)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 80)
    X1g, X2g = np.meshgrid(x1, x2)
    Z = y_true_func(X1g, X2g)
    ax.plot_surface(X1g, X2g, Z, cmap="viridis", alpha=0.5, linewidth=0, antialiased=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(1.0)
    plt.close(fig)


def generate_classification_data(seed=SEED, n_samples=500):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    angle = np.arctan2(X[:, 1], X[:, 0])
    y = ((angle + radius * 0.5) % (2 * np.pi) < np.pi).astype(int)
    noise_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    return X, y


def plot_classification(X, y, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="C0", label="class 0", alpha=0.8, edgecolor="k")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="C1", label="class 1", alpha=0.8, edgecolor="k")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(1.0)
    plt.close(fig)


def generate_logic_gate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gates = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "XOR": np.array([0, 1, 1, 0]),
    }
    return X, gates


def print_truth_tables(X, gates):
    header = "A B | AND OR NAND XOR"
    print(header)
    print("-" * len(header))
    for i, inp in enumerate(X):
        a, b = inp
        row = f"{a} {b} |  {gates['AND'][i]}   {gates['OR'][i]}    {gates['NAND'][i]}    {gates['XOR'][i]}"
        print(row)


def save_dataset_npz(filename, **kwargs):
    path = os.path.join(OUT_DIR, filename)
    np.savez_compressed(path, **kwargs)
    return path


def main():
    print(f"Using seed = {SEED}")
    # Regression
    X_reg, y_reg, y_reg_true = generate_regression_data(SEED, n_samples=500)
    print("Regression shapes:", X_reg.shape, y_reg.shape)
    print("Saving regression dataset to outputs/")
    save_dataset_npz("regression_data.npz", X=X_reg, y=y_reg, y_true=y_reg_true)
    y_true_func = lambda x1, x2: 2 * x1 ** 2 - 1.5 * x1 * x2 + 0.8 * x2 ** 2
    reg_path = os.path.join(OUT_DIR, "regression_3d.png")
    plot_regression_3d(X_reg, y_reg, y_true_func, save_path=reg_path)
    print(f"Regression figure saved to: {reg_path}")

    # Classification
    X_clf, y_clf = generate_classification_data(SEED, n_samples=500)
    print("Classification shapes:", X_clf.shape, y_clf.shape)
    unique, counts = np.unique(y_clf, return_counts=True)
    print("Class distribution:", dict(zip(unique.tolist(), counts.tolist())))
    print("Saving classification dataset to outputs/")
    save_dataset_npz("classification_data.npz", X=X_clf, y=y_clf)
    clf_path = os.path.join(OUT_DIR, "classification_scatter.png")
    plot_classification(X_clf, y_clf, save_path=clf_path)
    print(f"Classification figure saved to: {clf_path}")

    # Logic gates
    X_logic, gates = generate_logic_gate_data()
    print("\nLogic gate inputs:\n", X_logic)
    print("\nTruth tables:")
    print_truth_tables(X_logic, gates)

    # Final message and list saved files
    print("\nSaved files in 'outputs' directory:")
    for fname in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, fname))


if __name__ == "__main__":
    main()
