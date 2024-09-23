import tensorflow as tf
from qssl.config import Config
from qssl.loss.losses import Losses
from tensorflow.keras import layers, models, optimizers
import wandb
import torch

## CNN Encoder Trainer
class Trainer:
    def __init__(self, siamese_network, pairs_train, labels_train, pairs_test, labels_test):
        self.siamese_network = siamese_network
        self.pairs_train = pairs_train
        self.labels_train = labels_train
        self.pairs_test = pairs_test
        self.labels_test = labels_test

    def train(self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, learning_rate=Config.LEARNING_RATE):
        tf.get_logger().setLevel('ERROR')

        self.siamese_network.compile(
            loss=Losses.contrastive_pair_loss(),
            optimizer=optimizers.Adam(learning_rate=learning_rate)
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='qssl_hybrid_model.h5', save_weights_only=True, verbose=1)

        history = self.siamese_network.fit(
            [self.pairs_train[:, 0], self.pairs_train[:, 1]], self.labels_train,
            validation_data=([self.pairs_test[:, 0], self.pairs_test[:, 1]], self.labels_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[cp_callback]
        )
        return history
        
## GNN Training

def run_model(model, epoch, dataloader, lossFn, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    num_samples = 0

    for batch in dataloader:
        optimizer.zero_grad() if train else None

        # Extract the pairs and labels from the dataloader batch
        data1, data2, labels = batch[0], batch[1], batch[2]
        
        # Get the embeddings for both graphs
        emb1 = model(data1.x.float(), data1.edge_index, data1.batch)
        emb2 = model(data2.x.float(), data2.edge_index, data2.batch)
        
        # Compute contrastive loss
        loss = lossFn(emb1, emb2)  
        
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data1.num_graphs
        num_samples += data1.num_graphs

    avg_loss = total_loss / num_samples
    return avg_loss

def run_qmodel(model, epoch, dataloader, lossFn, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    num_samples = 0

    for batch in dataloader:
        optimizer.zero_grad() if train else None

        # Extract the pairs and labels from the dataloader batch
        data1, data2, labels = batch[0], batch[1], batch[2]
        
        # Get the embeddings for both graphs
        emb1 = model(data1.x.float(), data1.edge_index, data1.batch)
        emb2 = model(data2.x.float(), data2.edge_index, data2.batch)
        
        # Compute contrastive loss
        # label = torch.tensor([1 if data1.label == data2.label else 0])
        loss = lossFn(emb1, emb2,labels)  
        
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data1.num_graphs
        num_samples += data1.num_graphs

    avg_loss = total_loss / num_samples
    return avg_loss



def train_model(model, optimizer, lossFn, epochs, lr, train_dataloader, val_dataloader):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Run training
        train_loss = run_model(model, epoch, train_dataloader, lossFn, optimizer)
        
        # Run validation
        val_loss = run_model(model, epoch, val_dataloader, lossFn, optimizer=None, train=False)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Store loss for this epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log losses to WandB
        wandb.log({
            "qgnn epoch": epoch + 1,
            "qgnn train_loss": train_loss,
            "qgnn val_loss": val_loss
        })

    return history

def train_qmodel(model, optimizer, lossFn, epochs, lr, train_dataloader, val_dataloader):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Run training
        train_loss = run_qmodel(model, epoch, train_dataloader, lossFn, optimizer)
        
        # Run validation
        val_loss = run_qmodel(model, epoch, val_dataloader, lossFn, optimizer=None, train=False)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Store loss for this epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log losses to WandB
        wandb.log({
            "qgnn epoch": epoch + 1,
            "qgnn train_loss": train_loss,
            "qgnn val_loss": val_loss
        })

    return history