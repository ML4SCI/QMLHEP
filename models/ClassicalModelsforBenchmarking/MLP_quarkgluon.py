import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
import h5py

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_and_preprocess(file_path, Nsamples=1000, crop_size=12):
    with h5py.File(file_path, "r") as f:
        X_jets = f["X_jets"][:Nsamples]
        y = f["y"][:Nsamples]

    def crop_center(img, cx, cy):
        x, y = img.shape[1:3]
        sx, sy = x // 2 - cx // 2, y // 2 - cy // 2
        return img[:, sx:sx+cx, sy:sy+cy, :]

    cropped = crop_center(X_jets, crop_size, crop_size)
    ch0 = cropped[..., 1]
    ch1 = cropped[..., 2]

    def normalize(ch):
        div = np.max(ch, axis=(1, 2), keepdims=True)
        div[div == 0] = 1
        return ch / (div + 1e-5)

    X = np.concatenate([normalize(ch0), normalize(ch1)], axis=-1)
    X = X.reshape(Nsamples, -1)
    X = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1
    return train_test_split(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long),
                            test_size=0.2, stratify=y, random_state=42)
    
    

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    

def train_mlp_model(file_path, epochs=50, batch_size=32, lr=0.001):
    X_train, X_test, y_train, y_test = load_and_preprocess(file_path, Nsamples=1000, crop_size=12)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim=input_dim, hidden_dim=128, output_dim=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            test_correct, test_total = 0, 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                test_correct += (out.argmax(1) == yb).sum().item()
                test_total += yb.size(0)
        test_acc = test_correct / test_total

        print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

train_mlp_model("C:\\Users\\riakh\\Downloads\\quark-gluon_train-set_n793900-001.hdf5", epochs=70)