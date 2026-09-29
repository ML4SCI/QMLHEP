import time
import librosa
from pennylane import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# ========================================================
# 1. ENHANCED MULTI-CLASS 2D ACOUSTIC PATCH INGESTION
# ========================================================
files = [
    librosa.example('nutcracker'), 
    librosa.example('choice'),
    librosa.example('trumpet'),
    librosa.example('vibeace')
]

def extract_2d_spectrogram_patches(file_path):
    """Extracts a static 4x4 frequency-time block preserving structural layout."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=4, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    mid_frame = mel_db.shape[1] // 2
    patch_2d = mel_db[:, mid_frame:mid_frame + 4]
    
    if patch_2d.shape[1] < 4:
        patch_2d = np.pad(patch_2d, ((0, 0), (0, 4 - patch_2d.shape[1])), mode='constant')
        
    return patch_2d 

base_patches = [extract_2d_spectrogram_patches(f) for f in files]
num_classes = 4

def generate_multi_class_2d_dataset(size):
    """Generates a dataset split evenly across 4 distinct audio classes."""
    samples_per_class = size // num_classes
    X_list, y_list = [], []
    
    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            noisy_patch = base_patches[class_idx] + np.random.normal(0, 0.2, size=(4, 4))
            X_list.append(noisy_patch)
            y_list.append(class_idx)
            
    X_raw = np.array(X_list)
    y = np.array(y_list)
    
    X_flat = X_raw.reshape(size, 16)
    X_scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(np.array(X_flat))
    return X_scaled.reshape(size, 4, 4), y

dataset_size = 40
X_data, y_data = generate_multi_class_2d_dataset(dataset_size)

target_map = {
    0: np.array([0.8, 0.8]),
    1: np.array([0.8, -0.8]),
    2: np.array([-0.8, 0.8]),
    3: np.array([-0.8, -0.8])
}

y_transformed = np.array([target_map[int(val)] for val in y_data])

# ========================================================
# 2. HIERARCHICAL QCNN WITH INDEPENDENT PAIR WEIGHTS
# ========================================================
n_qubits = 8 
dev = qml.device("lightning.qubit", wires=n_qubits)

def quantum_conv_layer_unshared(params, wires):
    """Applies independent parameters to each qubit pair to break spatial symmetry."""
    param_idx = 0
    # Even pairs
    for w1, w2 in zip(wires[0::2], wires[1::2]):
        qml.CRX(params[param_idx], wires=[w1, w2])
        qml.CRY(params[param_idx + 1], wires=[w2, w1])
        param_idx += 2
    # Odd pairs
    for w1, w2 in zip(wires[1::2], wires[2::2]):
        qml.CRX(params[param_idx], wires=[w1, w2])
        param_idx += 1

def quantum_pooling_layer(wires_sink, wires_source):
    for si, so in zip(wires_sink, wires_source):
        qml.CRZ(np.pi / 2, wires=[so, si])

@qml.qnode(dev)
def qcnn_circuit(patch_2d, weights):
    features = patch_2d.flatten()
    for i in range(n_qubits):
        qml.RX(features[i], wires=i)
        qml.RY(features[i + 8], wires=i)
        
    # Layer 1: 8 wires -> 11 parameters
    quantum_conv_layer_unshared(weights[:11], wires=list(range(8)))
    quantum_pooling_layer(wires_sink=[0, 2, 4, 6], wires_source=[1, 3, 5, 7])
    
    # Layer 2: 4 active wires -> 5 parameters
    quantum_conv_layer_unshared(weights[11:16], wires=[0, 2, 4, 6])
    quantum_pooling_layer(wires_sink=[0, 4], wires_source=[2, 6])
    
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(4))]

# ========================================================
# 3. OPTIMIZATION & EVALUATION
# ========================================================
def cost_function(weights, X_batch, y_batch):
    loss = 0.0
    for sample, target in zip(X_batch, y_batch):
        expectations = qcnn_circuit(sample, weights)
        loss += (expectations[0] - target[0]) ** 2 + (expectations[1] - target[1]) ** 2
    return loss / len(X_batch)

initial_lr = 0.15
decay_rate = 0.85
decay_patience = 3
patience = 8

np.random.seed(42)
# Total of 16 independent parameters across conv blocks
weights = np.random.randn(16, requires_grad=True)

best_weights = weights.copy()
best_loss = float('inf')
patience_counter = 0

print("🏋️ Starting Asymmetric QCNN Parametric Training Loop...")
print("---------------------------------------------------------")

epochs = 35
batch_size = 8
current_lr = initial_lr

for epoch in range(epochs):
    if epoch > 0 and epoch % decay_patience == 0:
        current_lr *= decay_rate

    opt = qml.AdamOptimizer(stepsize=current_lr)

    indices = np.random.permutation(len(X_data))
    X_shuffled = X_data[indices]
    y_shuffled = y_transformed[indices]
    
    X_batch = X_shuffled[:batch_size]
    y_batch = y_shuffled[:batch_size]
    
    weights, current_loss = opt.step_and_cost(lambda w: cost_function(w, X_batch, y_batch), weights)
    
    if current_loss < best_loss:
        best_loss = current_loss
        best_weights = weights.copy()
        patience_counter = 0
        status_msg = "⭐ (New Best Saved)"
    else:
        patience_counter += 1
        status_msg = ""

    print(f"Epoch {epoch+1:02d} | LR: {current_lr:.4f} | Loss: {current_loss:.4f} {status_msg}")

    if patience_counter >= patience:
        print(f"\n🛑 Early stopping triggered at Epoch {epoch+1}!")
        break

print("---------------------------------------------------------")
print("🔮 Evaluation: Distance Decoding on Best Weights...")

def decode_expectations_euclidean(exp_vector):
    exp_arr = np.array(exp_vector)
    best_class = 0
    min_dist = float('inf')
    
    for class_idx, target_vec in target_map.items():
        dist = np.sum((exp_arr - target_vec) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_class = class_idx
            
    return best_class

final_predictions = []
for sample in X_data:
    raw_scores = qcnn_circuit(sample, best_weights)
    predicted_class = decode_expectations_euclidean(raw_scores)
    final_predictions.append(predicted_class)

y_true = [int(val) for val in y_data]
final_acc = accuracy_score(y_true, final_predictions)

print(f"\n🏆 Post-Training Classification Accuracy: {final_acc * 100:.2f}%\n")
print("📊 Detailed Classification Report:")
print(classification_report(y_true, final_predictions, target_names=['Nutcracker', 'Choice', 'Trumpet', 'Vibe Ace'], zero_division=0))