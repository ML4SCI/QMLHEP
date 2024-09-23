import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix as cmatrix
import wandb
import itertools
from ..evaluation.evaluate import plot_auc, make_cm

## Trainer and tester for ResNet18

class Trainer:

    ##########################
    #                        #
    #     Initialization     #
    #                        #
    ##########################



    def __init__(self,
                 model,
                 dataloader,
                 optimizer = None,
                 lr = 0.0005,
                 loss_function=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        # self.model = model.to(device)
        self.model = model
        self.dataloader = dataloader
        
        
        ## Initialize the optimizer if none has been passed in
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        else:
            self.optimizer = optimizer


        ## Initialize the loss function(s)
        self.loss_function = loss_function
        
        # self.device = device
        
        self.curr_epoch = 0

    ##########################
    #                        #
    #  Single Iter Training  #
    #                        #
    ##########################


    def train_iter(self, x1, x2, labels, verbose=0):

        ## Zero the gradients
        self.optimizer.zero_grad()

        ## Pass the inputs through the model
        emb1 = self.model(x1)
        emb2 = self.model(x2)


        ## Calculate the loss(es)
        loss = self.loss_function(emb1, emb2)
        

        ## Pass the loss backward
        loss.backward()

        ## Take an optimizer step
        self.optimizer.step()

        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
            predictions = (cos_sim > 0.5).long()  # You can adjust the threshold
            # correct = (predictions == labels.to(self.device)).sum().item()
            correct = (predictions == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total

        ## Return the total loss
        return(loss,accuracy)

    ##########################
    #                        #
    #  Mutlti Epoch Training #
    #                        #
    ##########################



    def train(self,
              epochs,
              print_every=1,
              writer=None):

        
        ## Loop over epochs in the range of epochs
        epoch_losses = []
        epoch_acc = []

        for epoch in range(self.curr_epoch, self.curr_epoch + epochs):
            

            ## If the report_every epoch is reached, reinitialize metric lists
            if epoch % print_every == 0:
                print("----- Epoch: " + str(epoch) + " -----")
            
    
            ## Enumerate self.dataloader
            #tot_iter = self.dataloader.dataset.__len__() // self.dataloader.batch_size
            
            #for idx, data_dict in enumerate(tqdm(self.dataloader, total=tot_iter)):
            batch_losses = []
            batch_acc = []

            for idx, data_dict in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                
                ## Grab an example
                x1 = data_dict["x1"]; x2 = data_dict["x2"]
                labels = data_dict["labels"]
                
                
                ## Send it to self.device
                # x1 = x1.to(self.device); x2 = x2.to(self.device)
                
                
                ## Try to train_iter
                batch_loss, batch_acc = self.train_iter(x1, x2, labels)


                ## Update the metric lists and counters
                batch_losses.append(batch_loss.item())
            
            
            epoch_losses.append(np.mean(batch_losses))
            epoch_acc.append(np.mean(batch_acc))
            self.curr_epoch += 1
            
    
            ## If we've hit report_every epoch, print the report
            if epoch % print_every == 0:
                print("Avg train loss: " + str(np.mean(epoch_losses)))
                print("Avg train accuracy: " + str(np.mean(epoch_acc)))


                ## Logging
                if writer is not None:
                    pass 
                
        return(epoch_losses, epoch_acc)
                
                
                
class Tester:

    ##########################
    #                        #
    #     Initialization     #
    #                        #
    ##########################



    def __init__(self,
                 model,
                 dataloader,
                 metric=None,
                 classes=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        # self.model = model.to(device)
        model.eval()
        self.model = model
        self.dataloader = dataloader
        self.metric = metric
        self.classes = classes
        self.device = device
        self.curr_epoch = 0

        
    ##########################
    #                        #
    #  Single Iter Testing   #
    #                        #
    ##########################


    def test_batch(self, input, label):

        ## Pass the inputs through the model
        x1, x2 = input
        emb1 = self.model(x1)
        emb2 = self.model(x2)

        
        # res = self.model.predict(x1,x2)
        
        # prediction = res.detach().cpu()

        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(emb1,emb2)
            prediction = (cos_sim > 0.5).long()

        label = label.detach().cpu()

        ## Calculate the metric
        # metric = self.metric(prediction.flatten(), label.flatten())

        correct = (prediction == label).sum().item()
        total = label.size(0)
        accuracy = correct / total

        ## Return the metric
        return(accuracy,label,prediction)

    ##########################
    #                        #
    #  Mutlti Epoch Testing  #
    #                        #
    ##########################



    def test(self):
        
        
        batch_metrics = []

        ## Enumerate self.dataloader
        for idx, data_dict in enumerate(self.dataloader):

            ## Grab an example
            x1 = data_dict["x1"]; x2 = data_dict["x2"]
            y = data_dict["labels"]

            ## Send it to self.device
            # x1 = x1.to(self.device); x2 = x2.to(self.device)
            # y = y.to(self.device)

            ## Try to train_iter
            batch_metric,label,prediction = self.test_batch((x1,x2), y)

            ## Update the metric lists and counters
            # batch_metrics.append(batch_metric.item())
            batch_metrics.append(batch_metric)
            acc = np.mean(batch_metrics)
        print("Test accuracy: ", 100. * acc)
            
        return(acc,label,prediction)
    
    def test_eval(self):
        model = self.model
        test_dataloader = self.dataloader
        preds = []
        labels = []
        acc = 0
        for data in test_dataloader:

            target = data.labels
            labels.append(target.detach().cpu().numpy())

            output = model(data.x, data.edge_index.type(torch.int64), data.batch)
            preds.append(output.detach().cpu().numpy())  # Convert to numpy array
            # probs = Sigmoid()(output).detach().cpu().numpy()  # Convert to numpy array
            # preds.append(copy.deepcopy(output))

            target = target.unsqueeze(1).float()
            pred_labels = (output > 0).float()
            acc += (pred_labels == target).sum().item()

        acc = acc / len(test_dataloader.dataset)
        print("Test accuracy: ", 100. * acc)

        # Concatenate lists into a single numpy array
        labels = np.concatenate(labels, axis=0)
        # Concatenate lists into a single numpy array
        preds = np.concatenate(preds, axis=0)

        return labels, preds


    def auc(self):
        _, labels, preds = self.test()
        plot_auc(labels, preds)

    def confusion_matrix(self):
        _, labels, preds = self.test()
        make_cm(labels,preds,classes=self.classes)
