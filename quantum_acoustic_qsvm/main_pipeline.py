import time
import librosa
import numpy as np
import pennylane as qml
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# 1. GENERATE BASE DATASET (DSP FUNNEL)
# ==========================================
files = [librosa.example('nutcracker'), librosa.example('choice')]

def extract_dsp_features(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return np.mean(mel_db, axis=1)

base_0 = extract_dsp_features(files[0])
base_1 = extract_dsp_features(files[1])

def generate_dataset(size):
    half = size // 2
    X = np.vstack([
        base_0 + np.random.normal(0, 0.5, size=(half, 32)),
        base_1 + np.random.normal(0, 0.5, size=(half, 32))
    ])
    y = np.array([0] * half + [1] * half)
    X_f = SelectKBest(score_func=f_classif, k=4).fit_transform(X, y)
    X_q = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_f)
    return X_q, y

# Standard operational dataset size
X_data, y_data = generate_dataset(60)

# ==========================================
# 2. DEFINE QUANTUM KERNEL (WITH ENTANGLEMENT)
# ==========================================
n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits)

def ansatz(x, wires):
    """Expressive quantum feature map with single-qubit rotations and entanglement."""
    for i in range(len(wires)):
        qml.RX(x[i], wires=wires[i])
        qml.RY(x[i], wires=wires[i])
    # Entangling layer
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])

@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):
    ansatz(x1, wires=range(n_qubits))
    qml.adjoint(ansatz)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

kernel_func = lambda x1, x2: quantum_kernel_circuit(x1, x2)[0]

# ==========================================
# 3. STRATIFIED CROSS-VALIDATION & HYPERPARAMETER TUNING
# ==========================================
print("📦 Computing Global Training Gram Matrix...")
global_kernel_matrix = qml.kernels.square_kernel_matrix(X_data, kernel_func)

param_grid = [0.1, 1.0, 10.0, 100.0]
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = 0.0
best_C = None

print("⚡ Running Hyperparameter Optimization across Quantum Features...")
for C in param_grid:
    fold_scores = []
    for train_idx, val_idx in cv_strategy.split(X_data, y_data):
        # Slice Gram matrix correctly for precomputed SVM kernel
        K_train = global_kernel_matrix[np.ix_(train_idx, train_idx)]
        K_val = global_kernel_matrix[np.ix_(val_idx, train_idx)]
        
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K_train, y_data[train_idx])
        preds = clf.predict(K_val)
        fold_scores.append(accuracy_score(y_data[val_idx], preds))
        
    mean_score = np.mean(fold_scores)
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

print(f"\n🏆 Grid Search Optimization Complete!")
print(f"Best Regularization Parameter ('C'): {best_C}")
print(f"Mean Validation Cross-Validation Accuracy Score: {best_score * 100:.2f}%\n")