from scipy.stats import ttest_ind
import numpy as np

X_flat = X.reshape(X.shape[0], -1)  
quark_flat = X_flat[y == 0]
gluon_flat = X_flat[y == 1]
t_vals, p_vals = ttest_ind(quark_flat, gluon_flat, axis=0, equal_var=False)

top_indices = np.argsort(p_vals)[:192]  # top 8x8x3 = 192 pixel indices

X_selected = X_flat[:, top_indices]  # shape: (1000, 192)


from sklearn.model_selection import train_test_split
import torch

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
