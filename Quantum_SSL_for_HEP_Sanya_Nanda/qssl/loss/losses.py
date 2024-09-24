import torch
import numpy as np
import torch.nn.functional as F
import tensorflow as tf
from qssl.config import Config
import pennylane as qml
import tensorflow as tf
from tensorflow import keras

def form_pairs(inA, inB):
    
    '''
    Form pairs from two tensors of embeddings. It is assumed that the embeddings at corresponding batch positions are similar
    and all other batch positions are dissimilar 
    '''
    
    b, emb_size = inA.shape
    perms = b**2
    
    labels = [0]*perms; sim_idxs = [(0 + i*b) + i for i in range(b)]
    for idx in sim_idxs:
        labels[idx] = 1
    labels = torch.Tensor(labels)
    
    return(inA.repeat(b, 1), torch.cat([inB[i,:].repeat(b,1) for i in range(b)]), labels.type(torch.LongTensor).to(inA.device))


class NTXent(torch.nn.Module):
    
    '''
    Modified from: https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
    '''
    
    def __init__(self, 
                 batch_size, 
                 temperature=0.5,
                 device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        self.device = device
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    
    def __init__(self,
                 distance = lambda x,y: torch.pow(x-y, 2).sum(1),
                 margin=1.0,
                 mode='pairs',
                 batch_size=None,
                 temperature=0.5):
        
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.mode = mode
        
        if self.mode == 'ntxent':
            assert batch_size is not None, "Must specify batch size to use Ntxent Loss"
            self.ntxent = NTXent(batch_size = batch_size, temperature = temperature)
        
    
    
    
    def forward(self, x, y):
        
        if self.mode == 'pairs':
            return(self.forward_pairs(x, y))
        
        elif self.mode == 'ntxent':
            return(self.forward_ntxent(x, y))
    
    
    def forward_ntxent(self, x, y):
        return(self.ntxent(x, y))
    
    def forward_pairs(self, x, y, label=None):
        '''
        Return the contrastive loss between two similar or dissimilar outputs
        '''
        
        assert x.shape==y.shape, str(x.shape) + "does not match input 2: " + str(y.shape)
        
        x, y, label = form_pairs(x,y)
        
        distance = self.distance(x,y)
        
        # When the label is 1 (similar) - the loss is the distance between the embeddings
        # When the label is 0 (dissimilar) - the loss is the distance between the embeddings and a margin
        loss_contrastive = torch.mean((label) * distance +
                                      (1-label) * torch.clamp(self.margin - distance, min=0.0))


        return loss_contrastive


class Losses:
    @staticmethod
    def quantum_fidelity_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def contrastive_pair_loss(margin=Config.MARGIN):
        def loss(y_true, dist):
            y_true = tf.cast(y_true, tf.float32)
            square_dist = tf.square(dist)
            margin_square = tf.square(tf.maximum(margin - dist, 0))
            return tf.reduce_mean(y_true * square_dist + (1 - y_true) * margin_square)
        return loss


class InfoNCELoss(keras.losses.Loss):
    def __init__(self, n_qubits, n_ancillas, q_depth, q_params, temperature=0.1, epsilon=1e-4, negative_mode='unpaired'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.negative_mode = negative_mode
        self.q_params = q_params
        self.q_depth = q_depth
        self.n_qubits = n_qubits
        self.n_ancillas = n_ancillas

    def call(self, query, positive_key, negative_keys):
        q_in_query = tf.tanh(query) * np.pi / 2.0
        q_in_pos = tf.tanh(positive_key) * np.pi / 2.0

        positive_logit = self.compute_similarity(q_in_query, q_in_pos)

        if self.negative_mode == 'unpaired':
            q_in_neg = tf.tanh(negative_keys) * np.pi / 2.0
            negative_logits = self.compute_similarity(q_in_query, q_in_neg)
        else:
            q_in_neg = tf.tanh(negative_keys) * np.pi / 2.0
            negative_logits = self.compute_similarity(q_in_query, q_in_neg, mode='paired')

        logits = tf.concat([positive_logit, negative_logits], axis=1)
        labels = tf.zeros(len(logits), dtype=tf.int64)
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits / self.temperature, from_logits=True)

    def compute_similarity(self, query, key, mode='paired'):
        if mode == 'paired':
            similarity = []
            for h1, h2 in zip(query, key):
                q_pair = tf.concat([h1, h2], axis=0)
                sim = quantum_circuit(q_pair, self.q_params, self.q_depth, self.n_qubits, self.n_ancillas, training=True)
                similarity.append(tf.reduce_sum(sim)**2)
            return tf.stack(similarity, axis=0)[:, tf.newaxis]
        else:
            similarity = []
            for h_query in query:
                row_aux = []
                for h_key in key:
                    q_pair = tf.concat([h_query, h_key], axis=0)
                    sim = quantum_circuit(q_pair, self.q_params, self.q_depth, self.n_qubits, self.n_ancillas, training=True)
                    row_aux.append(tf.reduce_sum(sim)**2)
                similarity.append(tf.stack(row_aux, axis=0))
            return tf.stack(similarity, axis=0)