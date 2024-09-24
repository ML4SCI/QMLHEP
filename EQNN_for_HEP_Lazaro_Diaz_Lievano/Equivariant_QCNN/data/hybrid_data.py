
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

def load_hybrid_data(dataset, train_size, test_size, classes = [0,1]):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),  
        transforms.ToTensor()        
    ])

    if dataset == "mnist":
        
        fashion_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        fashion_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_indices = [i for i, (x, y) in enumerate(fashion_trainset) if y in classes]
        test_indices = [i for i, (x, y) in enumerate(fashion_testset) if y in classes]

        train_indices = train_indices[:train_size]
        test_indices = test_indices[:test_size]

        fashion_trainset_small = Subset(fashion_trainset, train_indices)
        fashion_testset_small = Subset(fashion_testset, test_indices)

        X_train = torch.stack([item[0] for item in fashion_trainset_small])
        y_train = torch.tensor([item[1] for item in fashion_trainset_small])

        X_test = torch.stack([item[0] for item in fashion_testset_small])
        y_test = torch.tensor([item[1] for item in fashion_testset_small])

        X_train = X_train.reshape(train_size, 16, 16, 1)
        X_test = X_test.reshape(test_size, 16, 16, 1)

    elif dataset == "fashion_mnist":
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import Subset

        fashion_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        fashion_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        train_indices = [i for i, (x, y) in enumerate(fashion_trainset) if y in classes]
        test_indices = [i for i, (x, y) in enumerate(fashion_testset) if y in classes]

        train_indices = train_indices[:train_size]
        test_indices = test_indices[:test_size]

        fashion_trainset_small = Subset(fashion_trainset, train_indices)
        fashion_testset_small = Subset(fashion_testset, test_indices)

        X_train = torch.stack([item[0] for item in fashion_trainset_small])
        y_train = torch.tensor([item[1] for item in fashion_trainset_small])

        X_test = torch.stack([item[0] for item in fashion_testset_small])
        y_test = torch.tensor([item[1] for item in fashion_testset_small])

        X_train = X_train.reshape(train_size, 16, 16, 1)
        X_test = X_test.reshape(test_size, 16, 16, 1)

    elif dataset == "electron_photon":
        import numpy as np
        import h5py
        import torch
        from sklearn.model_selection import train_test_split
        from torchvision import transforms

        path_ep = "/home/lazaror/quantum/pruebas/EQCNN_local_testing/EQNN_for_HEP/Equivariant_QCNN/data/E-P_rescaled"
        with h5py.File(path_ep, "r") as file:
            X_ep = np.array(file["X"])
            y_ep = np.array(file["y"])

            X_train_16, X_test_16, Y_train_16, Y_test_16 = train_test_split(X_ep, y_ep, test_size=0.2, random_state=42, stratify=y_ep)

            transform = transforms.Compose([
                transforms.ToPILImage(),          
                transforms.Resize((16, 16)),      
                transforms.ToTensor()             
            ])

            X_train_resized = torch.stack([transform(x) for x in X_train_16])
            X_test_resized = torch.stack([transform(x) for x in X_test_16])

            X_train = X_train_resized.permute(0, 2, 3, 1)  
            X_test = X_test_resized.permute(0, 2, 3, 1)   

            y_train = torch.tensor(Y_train_16)
            y_test = torch.tensor(Y_test_16)

    elif dataset == "quark_gluon":
        import numpy as np
        import h5py
        import torch
        from sklearn.model_selection import train_test_split
        from torchvision import transforms

        path_ep = "/home/lazaror/quantum/pruebas/EQCNN_local_testing/EQNN_for_HEP/Equivariant_QCNN/data/E-P_rescaled"
        with h5py.File(path_ep, "r") as file:
            X_ep = np.array(file["X"])
            y_ep = np.array(file["y"])

            X_train_16, X_test_16, Y_train_16, Y_test_16 = train_test_split(X_ep, y_ep, test_size=0.2, random_state=42, stratify=y_ep)

            transform = transforms.Compose([
                transforms.ToPILImage(),          
                transforms.Resize((16, 16)),      
                transforms.ToTensor()             
            ])

            X_train_resized = torch.stack([transform(x) for x in X_train_16])
            X_test_resized = torch.stack([transform(x) for x in X_test_16])

            X_train = X_train_resized.permute(0, 2, 3, 1) 
            X_test = X_test_resized.permute(0, 2, 3, 1)    

            y_train = torch.tensor(Y_train_16)
            y_test = torch.tensor(Y_test_16)

    else:
        print("The dataset is not already implemented. Try using 'mnist', 'fashion_mnist', 'electron_photon' or 'quark_gluon'" )

    return X_train, y_train, X_test, y_test