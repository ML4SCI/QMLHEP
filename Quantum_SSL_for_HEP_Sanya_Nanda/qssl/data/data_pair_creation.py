# Required Imports
import numpy as np

# loading the data as train and test
data = np.load('../../data/electron-photon-large.npz', allow_pickle=True)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

print(f"Data Loading: x_train shape {x_train.shape}, x_test shape: {x_test.shape}")
print(f"Data Loading: y_train shape {y_train.shape}, y_test shape: {y_test.shape}")

# Preprocess the dataset
def preprocess_data(images, labels):
    # Add a dimenison for channel
    images = np.expand_dims(images, -1)
    # Normalize
    images = images.astype('float32') / 255.0  
    return images, labels


def crop(images, size):
    x = np.argmax(np.mean(images[:, :, :, 0], axis=0))
    center = [int(x/size), x%size]
    img_size = 8
    images = images[:, (center[0]-int(img_size/2)):(center[0]+int(img_size/2)), (center[1]-int(img_size/2)):(center[1]+int(img_size/2))]
    return images

x_train = x_train[:,:,:,0]
x_test = x_test[:,:,:,0]
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
x_train = crop(x_train, 32)
x_test = crop(x_test, 32)

print(f"After Data Preprocessing: x_train shape {x_train.shape}, x_test shape: {x_test.shape}")
print(f"After Data Preprocessing: y_train shape {y_train.shape}, y_test shape: {y_test.shape}")

# Create pairs of images and labels
def create_pairs(images, labels):
    pairs = []
    pair_labels = []

    num_classes = len(np.unique(labels)) # 2
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    # print(digit_indices)

    for idx1 in range(len(images)):
        x1, label1 = images[idx1], labels[idx1]
        idx2 = np.random.choice(digit_indices[int(label1)])
        x2 = images[idx2]

        # if x1==x2, label set to 1
        pairs.append([x1, x2])
        pair_labels.append(1)

        # if x1!=x2, label set to 0
        label2 = (label1 + np.random.randint(1, num_classes)) % num_classes
        idx2 = np.random.choice(digit_indices[int(label2)])
        x2 = images[idx2]
        pairs.append([x1, x2])
        pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)

pairs_train, labels_train = create_pairs(x_train, y_train)
pairs_test, labels_test = create_pairs(x_test, y_test)

print(f"After pair creation: pairs_train: {pairs_train.shape}, pairs_test: {pairs_test.shape}")
print(f"After pair creation: labels_train: {labels_train.shape}, labels_test: {labels_test.shape}")

np.savez_compressed('../../data/electron-photon-pairs.npz', **{
    'pairs_train': pairs_train,
    'labels_train':labels_train,
    'pairs_test':pairs_test,
    'labels_test':labels_test
})

# OUTPUT:
# Data Loading: x_train shape (398400, 32, 32, 2), x_test shape: (99600, 32, 32, 2)
# Data Loading: y_train shape (398400,), y_test shape: (99600,)
# After Data Preprocessing: x_train shape (398400, 8, 8, 1), x_test shape: (99600, 8, 8, 1)
# After Data Preprocessing: y_train shape (398400,), y_test shape: (99600,)
# After pair creation: pairs_train: (796800, 2, 8, 8, 1), pairs_test: (199200, 2, 8, 8, 1)
# After pair creation: labels_train: (796800,), labels_test: (199200,)


