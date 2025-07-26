import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from HybridQKAN_model_components import QSVT, quantum_lcu_block, QuantumSumBlock, KANLayer




def bump(x): return torch.exp(-10 * (x - 0.2)**2) + torch.exp(-50 * (x + 0.5)**2)
def runge(x): return 1 / (1 + 25 * x**2)
def exp_sin(x): return torch.exp(-x**2) * torch.sin(5 * x)
def noisy_step(x): return torch.heaviside(x, torch.tensor(0.0)) + 0.1 * torch.sin(20 * x)
def sigmoid_bumps(x): return torch.sigmoid(8 * (x - 0.5)) + torch.sigmoid(-10 * (x + 0.3)) - 1
def sawtooth(x): return 2 * (x - torch.floor(x + 0.5))
def default_func(x): return torch.tanh(10 * x + 0.5 + torch.clamp(x**2, min=0) * 10)

FUNCTION_MAP = {
    "bump": bump,
    "runge": runge,
    "exp_sin": exp_sin,
    "noisy_step": noisy_step,
    "sigmoid_bumps": sigmoid_bumps,
    "sawtooth": sawtooth,
    "default": lambda x: torch.tanh(10 * x + 0.5 + torch.clamp(x**2, min=0) * 10)
}


class QuantumKANRegressor(nn.Module):
    def __init__(self, num_features, degree=5):
        super().__init__()
        self.num_features = num_features
        self.degree = degree

        self.qsvt = QSVT(wires=1, degree=degree, depth=2)
        self.lcu_weights = nn.Parameter(torch.rand(num_features, degree))  # (F, P)
        self.sum_blocks = nn.ModuleList([QuantumSumBlock(degree) for _ in range(num_features)])
        self.kan = KANLayer(in_features=num_features, out_features=1)

    def forward(self, X):
        """
        Input: X of shape (B, F)
        Output: y_hat of shape (B, 1)
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
        return self.kan(features)
    
    
model = QuantumKANRegressor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
train_rmse, test_rmse = [], []
num_epochs = 1000

for name, func in FUNCTION_MAP.items():
    print(f"\nTraining on Function: {name}")

    x = torch.linspace(-1, 1, 500).unsqueeze(1)
    y = func(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            test_loss = criterion(test_outputs, y_test.to(device))

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

  
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        true = y_test.cpu().numpy()
        x_plot = X_test.cpu().numpy().squeeze()
        sort_idx = x_plot.argsort()

    plt.figure(figsize=(12, 5))

    
    plt.subplot(1, 2, 1)
    plt.plot(x_plot[sort_idx], true[sort_idx], label='Ground Truth', color='blue')
    plt.plot(x_plot[sort_idx], preds[sort_idx], '--', label='Prediction', color='red')
    plt.title(f"{name} – Prediction vs Ground Truth")
    plt.xlabel("Input x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_CURVE_results.png")
    plt.close()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f"{name} – Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_LOSS_results.png")
    plt.close()