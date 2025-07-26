import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from HybridQKAN_model_components import QSVT, quantum_lcu_block, QuantumSumBlock, KANLayer

df = pd.read_csv("C://Users//riakh//Downloads//archive//Titanic-Dataset.csv")

df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age', 'Fare']].values
y = df['Survived'].values

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

class QuantumKANClassifier(nn.Module):
    def __init__(self, num_features=4, degree=3, num_classes=2):
        super().__init__()
        self.qsvt = QSVT(wires=1, degree=degree, depth=2)
        self.lcu_weights = nn.Parameter(torch.rand(num_features, degree))
        self.sum_blocks = nn.ModuleList([QuantumSumBlock(degree) for _ in range(num_features)])
        self.kan = KANLayer(in_features=num_features, out_features=num_classes)

    def forward(self, X):
        B = X.size(0)
        features = []
        for i in range(B):
            xi = X[i]
            qsvt_vecs = [self.qsvt(xi[f]) for f in range(xi.shape[0])]
            lcu_vals = [quantum_lcu_block(qsvt_vecs[f], self.lcu_weights[f]) for f in range(len(qsvt_vecs))]
            summed = [self.sum_blocks[f](lcu_vals[f].unsqueeze(0)) for f in range(len(qsvt_vecs))]
            features.append(torch.stack(summed))
        return self.kan(torch.stack(features))

model = QuantumKANClassifier(num_features=4, degree=3, num_classes=2)
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
