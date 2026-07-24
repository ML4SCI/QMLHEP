import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import wandb
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



def extract_embeddings(test_dataloader, model, N, reduce_to_dimension=2, device='cuda'):
    
    '''
    Use a test dataloader and torch model to extract N embeddings, reduce them to 
    reduce_to_dimension dimensions, then organize them into a dataframe with the GT labels
    '''
        
    model.eval()
    m = model
    # m = model.to(device)
    
    embs = None
    labs = None
    tot_cnt = 0
    
    for idx, dat in enumerate(test_dataloader):
        
        # Extract relevent data points from batch
        x1 = dat['x1'] 
        x2 = dat['x2'] 
        labels = dat['labels'].type(torch.LongTensor) 
        
        # Pass inputs through model
        emb1 = m(x1); emb2 = m(x2)
        
        # Add to running lists
        if embs is None:
            embs = torch.cat([emb1, emb2])
        else:
            embs = torch.cat([embs, emb1, emb2])
        
        # Add coloring to list
        if labs is None:
            labs = torch.cat([labels, labels])
        else:
            labs = torch.cat([labs, labels, labels])

        # Check count of points so far
        curr_cnt = x1.shape[0] + x2.shape[0]
        if tot_cnt + curr_cnt >= N:
            break
        tot_cnt += curr_cnt
    
    # If there's only two dimensions in the embeddings we can directly use them
    embs = [np.array(e.cpu().detach()) for e in embs]
    emb_size = embs[0].shape[0]
    
    assert reduce_to_dimension is not None and reduce_to_dimension <= emb_size, "reduce_to_dimension must be integer <= emb dimension"
    if emb_size > reduce_to_dimension:
        from sklearn.decomposition import PCA
        embs = np.array(embs)
        print("performing PCA to reduce embeddings to " + str(reduce_to_dimension) + " dimensions")
        pca = PCA(n_components = reduce_to_dimension)
        embs = pca.fit_transform(embs)
        print(str(pca.explained_variance_ratio_.sum()) + " % variance explained using PCA")

    labs = [x.item() for x in labs]
    df = pd.DataFrame({"Emb": list(embs), "Label": labs})    
    return(df)


def plot_embeddings(emb_df):
    '''
    Plot the DataFrame from extract_embeddings() in 2 dimensions 
    and color by label
    '''
    embs = list(emb_df['Emb'])
    assert embs[0].shape[0] == 2, "Embeddings must be reduced to dimension 2, use reduce_to_dimension param in extract_embeddings"
    
    embs_x = [e[0] for e in embs]; embs_y = [e[1] for e in embs]
    labs = list(emb_df['Label'])
    
    import seaborn as sns
    sns.set_style("darkgrid")
    sns.relplot(x=embs_x, y=embs_y, hue=labs, palette="deep", alpha=0.7, s=75)
    plt.title("Test Embeddings")
    wandb.log({"Test Embeddings": wandb.Image(plt)})


def plot_auc(labels, preds):
    '''Plotas auc and logs to wandb'''
    auc = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.plot(fpr, tpr, label="AUC = {0}".format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    wandb.log({"AUC": wandb.Image(plt)})
    plt.show()


def visualize_graph_pairs(pairs, labels, num_pairs=3):
    '''visualise graph views'''
    plt.figure(figsize=(10, 5 * num_pairs))
    
    for i, (data1, data2) in enumerate(pairs[:num_pairs]):
        # Create a subplot for each pair
        plt.subplot(num_pairs, 2, 2 * i + 1)
        
        # Convert both graphs to NetworkX format for visualization
        G1 = to_networkx(data1, to_undirected=True)
        node_labels_1 = {j: f"pt: {data1.x[j][0]:.2f}, y: {data1.x[j][1]:.2f}, phi: {data1.x[j][2]:.2f}" 
                         for j in range(data1.x.size(0))}
        
        nx.draw(G1, with_labels=True, labels=node_labels_1, node_size=700, font_size=8)
        plt.title(f"Graph 1 (Label: {data1.y.item()})")

        plt.subplot(num_pairs, 2, 2 * i + 2)
        G2 = to_networkx(data2, to_undirected=True)
        node_labels_2 = {j: f"pt: {data2.x[j][0]:.2f}, y: {data2.x[j][1]:.2f}, phi: {data2.x[j][2]:.2f}" 
                         for j in range(data2.x.size(0))}
        
        nx.draw(G2, with_labels=True, labels=node_labels_2, node_size=700, font_size=8)
        plt.title(f"Graph 2 (Label: {data2.y.item()})")

        pair_type = "Positive Pair" if labels[i] == 1 else "Negative Pair"
        plt.suptitle(pair_type, fontsize=14, color='red')

    plt.tight_layout()
    plt.show()


def visualize_graph_pairs_01(pairs, labels):
    '''Visualise graph positive and negative view'''
    plt.figure(figsize=(10, 10))

    pos_pair_found = False
    neg_pair_found = False
    
    for i, (data1, data2) in enumerate(pairs):
        if labels[i] == 1 and not pos_pair_found:
            # Plot the positive pair
            plt.subplot(2, 2, 1)
            G1 = to_networkx(data1, to_undirected=True)
            node_labels_1 = {j: f"pt: {data1.x[j][0]:.2f}, y: {data1.x[j][1]:.2f}, phi: {data1.x[j][2]:.2f}" 
                             for j in range(data1.x.size(0))}
            nx.draw(G1, with_labels=True, labels=node_labels_1, node_size=700, font_size=8)
            plt.title(f"Positive Pair - Graph 1 (Label: {data1.y.item()})")

            plt.subplot(2, 2, 2)
            G2 = to_networkx(data2, to_undirected=True)
            node_labels_2 = {j: f"pt: {data2.x[j][0]:.2f}, y: {data2.x[j][1]:.2f}, phi: {data2.x[j][2]:.2f}" 
                             for j in range(data2.x.size(0))}
            nx.draw(G2, with_labels=True, labels=node_labels_2, node_size=700, font_size=8)
            plt.title(f"Positive Pair - Graph 2 (Label: {data2.y.item()})")
            
            pos_pair_found = True
        
        elif labels[i] == 0 and not neg_pair_found:
            # Plot the negative pair
            plt.subplot(2, 2, 3)
            G1 = to_networkx(data1, to_undirected=True)
            node_labels_1 = {j: f"pt: {data1.x[j][0]:.2f}, y: {data1.x[j][1]:.2f}, phi: {data1.x[j][2]:.2f}" 
                             for j in range(data1.x.size(0))}
            nx.draw(G1, with_labels=True, labels=node_labels_1, node_size=700, font_size=8)
            plt.title(f"Negative Pair - Graph 1 (Label: {data1.y.item()})")

            plt.subplot(2, 2, 4)
            G2 = to_networkx(data2, to_undirected=True)
            node_labels_2 = {j: f"pt: {data2.x[j][0]:.2f}, y: {data2.x[j][1]:.2f}, phi: {data2.x[j][2]:.2f}" 
                             for j in range(data2.x.size(0))}
            nx.draw(G2, with_labels=True, labels=node_labels_2, node_size=700, font_size=8)
            plt.title(f"Negative Pair - Graph 2 (Label: {data2.y.item()})")
            
            neg_pair_found = True

        if pos_pair_found and neg_pair_found:
            break

    plt.tight_layout()
    plt.savefig("pairs.png")    
    # Log the plot to WandB
    wandb.log({"Quark Gluon Pairs": wandb.Image("pairs.png")})
    plt.show()

def plot_and_save_loss(history):
    '''Learning History Plot'''
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss", color='blue')
    plt.plot(history["val_loss"], label="Validation Loss", color='orange')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig("loss_plot.png")
    
    # Log the plot to WandB
    wandb.log({"Learning History": wandb.Image("loss_plot.png")})
    
    plt.show()


def plot_auc(model, dataloader):
    '''Plots AUC'''
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_scores = []

    with torch.no_grad():
        for data1, data2, labels in dataloader:
            emb1 = model(data1.x, data1.edge_index, data1.batch)
            emb2 = model(data2.x, data2.edge_index, data2.batch)
            distances = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1)).cpu().numpy()  # L2 distance
            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
            # predictions = (cos_sim > 0.5).long()
            y_scores.extend(cos_sim)
            y_true.extend(labels.cpu().numpy())

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot AUC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(model, dataloader):
    '''Plots confusion matrix'''
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data1, data2, labels in dataloader:
            emb1 = model(data1.x, data1.edge_index, data1.batch)
            emb2 = model(data2.x, data2.edge_index, data2.batch)
            # distances = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1))  # L2 distance
            # predictions = (distances < 0.5).cpu().numpy()  # Threshold of 0.5 for similarity

            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
            predictions = (cos_sim > 0.5).long()
            
            y_pred.extend(predictions)
            y_true.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dissimilar", "Similar"])
    
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def plot_embeddings(model, dataloader):
    '''Plots embeddings'''
    model.eval()  # Set the model to evaluation mode
    embeddings = []
    labels = []

    with torch.no_grad():
        for data1, data2, label in dataloader:
            emb1 = model(data1.x, data1.edge_index, data1.batch)
            emb2 = model(data2.x, data2.edge_index, data2.batch)
            embeddings.extend(emb1.cpu().numpy())
            embeddings.extend(emb2.cpu().numpy())
            labels.extend(label.cpu().numpy())
            labels.extend(label.cpu().numpy())  # Add labels for both graphs in the pair

    # Reduce dimensions for visualization (e.g., t-SNE)
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
    
    # Plot embeddings
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm', s=10)
    plt.title("Graph Embeddings (t-SNE)")
    plt.colorbar()
    plt.show()



