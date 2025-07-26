import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from HybridQKAN_model_components import QSVT, quantum_lcu_block, QuantumSumBlock, KANLayer
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler


chebyshev_polynomials = [
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8]
]

iris = load_iris()
X = iris.data
y = iris.target
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

class QuantumKANClassifier(nn.Module):
    def __init__(self, num_features, degree=5, num_classes=2):
        super().__init__()
        self.num_features = num_features
        self.degree = degree

        self.qsvt = QSVT(wires=1, degree=degree, depth=2)
        self.lcu_weights = nn.Parameter(torch.rand(num_features, degree))  # (F, P)
        self.sum_blocks = nn.ModuleList([QuantumSumBlock(degree) for _ in range(num_features)])
        self.kan = KANLayer(in_features=num_features, out_features=num_classes)  # e.g., binary => 2

    def forward(self, X):
        """
        Input: X of shape (B, F)
        Output: logits of shape (B, num_classes)
        """
        B = X.size(0)
        all_features = []

        for i in range(B):
            xi = X[i]  # shape: (F,)
            qsvt_vecs = [self.qsvt(xi[f]) for f in range(self.num_features)]  # each: (P,)
            lcu_vals = [quantum_lcu_block(qsvt_vecs[f], self.lcu_weights[f]) for f in range(self.num_features)]
            summed = [self.sum_blocks[f](lcu_vals[f]) for f in range(self.num_features)]
            all_features.append(torch.stack(summed))  # shape: (F,)

        features = torch.stack(all_features)  # shape: (B, F)
        return self.kan(features)  # shape: (B, num_classes)

model = QuantumKANClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train = X_train.float().to(device)
y_train = y_train.long().to(device)
X_test = X_test.float().to(device)
y_test = y_test.long().to(device)

print("X_train dtype:", X_train.dtype)
print("weights dtype:", model.kan_layer.weights.dtype)
print("knots dtype:", model.kan_layer.knots.dtype)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred_train = outputs.argmax(dim=1)
        acc_train = (pred_train == y_train).float().mean()

    print(f"Epoch {epoch+1:02d} | Train Loss: {loss.item():.4f} | Train Acc: {acc_train.item():.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    pred_test = test_outputs.argmax(dim=1)
    acc_test = (pred_test == y_test).float().mean()
    print(f"\nTest Accuracy: {acc_test.item():.4f}")