import sys
import os
import torch
import random
import math
import time 

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

import torch.nn.functional as F

def evaluate( nodes_layers_dic, dataCenter, ds, GraphSage, Outlayer, device, max_vali_f1,  cur_epoch):
    test_f1 = 0
    test_nodes = getattr(dataCenter, ds+'_test')
    val_nodes = getattr(dataCenter, ds+'_val')
    labels = getattr(dataCenter, ds+'_labels')
    features = getattr(dataCenter, ds+'_feats')

    models = [GraphSage, Outlayer]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    embs = GraphSage( nodes_layers_dic,features[val_nodes])
    logists =  Outlayer(embs)
    predicts = torch.max(logists, 1)[1]
    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data) #???? why .data

    vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
    print("Validation F1:", vali_f1)

    if vali_f1 > max_vali_f1:
        max_vali_f1 = vali_f1
        embs = GraphSage( nodes_layers_dic, features[test_nodes])
        logists = Outlayer(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        print("Test F1:", test_f1)

        for param in params:
            param.requires_grad = True 

    for param in params:
        param.requires_grad = True

    return max_vali_f1, test_f1

def get_gnn_embeddings(nodes_layers_dic, gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    #features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.hidden_dim[-1]))
    features = getattr(dataCenter, ds+'_feats')
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = min(100, len(nodes))
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_layers_dic,features[nodes_batch])
        assert  (embs_batch.shape[0]) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def train_classification( nodes_layers_dic,dataCenter, GraphSage, Outlayer, ds, device, max_vali_f1,  b_sz = 100, epochs=10  ):
    print('Training Classification ...')
    c_optimizer = torch.optim.Adam(Outlayer.parameters(), lr=0.001, weight_decay = 5e-4)#torch.optim.SGD(Outlayer.parameters(), lr=0.1)
    # train classification, detached from the current graph
    #Outlayer.init_params()
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')
    features = getattr(dataCenter, ds+'_feats')
    embeddings = get_gnn_embeddings(nodes_layers_dic, GraphSage, dataCenter, ds)
    batches = math.ceil(len(train_nodes) / b_sz)
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = embeddings[nodes_batch]

            logists = Outlayer(embs_batch)
            labels_hot = F.one_hot(labels_batch, logists.shape[1])
            logists = torch.reshape(logists, [logists.shape[0], logists.shape[1]])
            loss = - torch.sum(logists * labels_hot)
            loss /= (logists.shape[0]) 
            loss.backward()
            
            nn.utils.clip_grad_norm_(Outlayer.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}]'.format(epoch+1,epochs, index, batches, loss.item(), len(visited_nodes),len(train_nodes)))
    max_vali_f1, test_f1 = evaluate( nodes_layers_dic,  dataCenter, ds, GraphSage, Outlayer, device, max_vali_f1,   epoch)
    return Outlayer, max_vali_f1, test_f1

 

def apply_model( optimizer, nodes_layers_dic,   dataCenter, ds, graphSage, classification, b_sz,      device, learn_method ):
    test_nodes = getattr(dataCenter, ds+'_test')
    val_nodes = getattr(dataCenter, ds+'_val')
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')
    features = getattr(dataCenter, ds+'_feats') 
  
    train_nodes = shuffle(train_nodes)

    models = [graphSage, classification]
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    for index in range(batches):
        t = time.time()
        nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
        visited_nodes |= set(nodes_batch)
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        features_batch = features[nodes_batch]
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch = graphSage( nodes_layers_dic, features_batch) 
        logists = classification(embs_batch) # b x 128 x 128
        batch_size = logists.size(0)
        num_classes = logists.size(1)
        labels_batch = F.one_hot(labels_batch, num_classes)
        logists = torch.reshape(logists, [batch_size, num_classes])
        loss_sup = - torch.sum(logists * labels_batch )
        loss_sup /= (len(nodes_batch) )#* num_classes
        loss = loss_sup 
        if index % (5 ) ==0:
            print(' Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(  index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step() 
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
        #print('time for one iter', time.time() - t)
    return graphSage, classification, optimizer
 
