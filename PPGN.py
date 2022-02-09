from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module 
import torch.nn as nn
import torch
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score, accuracy_score 
import sklearn.metrics
import numpy as np  
import torch.nn.functional as F  
import time 
import os, scipy
import torch
import scipy.sparse as sp
import numpy as np
import pickle
import torch
  

def one_hot ( Y  ):
    num_sample = Y.shape[0]
    num_labels = len(list(set(Y)))
    labels = np.zeros((num_sample, np.max(Y) + 1))
    for i in range(num_sample):
        labels[i,int(Y[i])] = 1 
    return labels
 

def get_y_map(n, sets):
    y_map = {}
    for i in range(n):
        y_map[i] = i
    change_list = []
    for s in sets:
        for v in s[1:]:
            y_map[v] = s[0]
            change_list.append(v)
    change_list = sorted(change_list)
    for i in range(n):
        if i not in change_list:
            n_change_pre = 0
            for change in change_list:
                if change < y_map[i]:
                    n_change_pre += 1
            y_map[i] -= n_change_pre
        else:
            y_map[i] = y_map[y_map[i]]
    return y_map  

def get_neib(  nodes, adj, prob,num_sample = 7):  
    _set = set
    to_neibs = [ adj[int(node)] for node in nodes]
    prob_node =  [ prob[[int(node)]*len(adj[int(node)]), adj[int(node)]] for node in nodes]
    if not num_sample is None:
        sample_neib = [set(np.random.choice(  to_neibs[ind_neib] , num_sample, p =  (prob_node[ind_neib]), replace = False )) if len( to_neibs[ind_neib] ) >= num_sample else  set(to_neibs[ind_neib]) for ind_neib in range(len(to_neibs)) ]
    else:
        sample_neib = to_neibs
    sample_neib = [sample | set([nodes[i]]) for i , sample in enumerate(sample_neib)   ] 
    unique_nodes_list = list(set.union(*sample_neib))  
    i = list(range(len(unique_nodes_list))) 
    unique_dic = dict(list(zip(unique_nodes_list, i))) 
    return unique_nodes_list, sample_neib, unique_dic

def dic_nodes_neib(num_layers,adj_list, prob, num_sample = 7):
    src_nodes  =list(range(prob.shape[0]))  
    nodes_layers = [(src_nodes,)] 
    for j in range(num_layers ):
        uniq_nodes, neib_nodes, uniq_dic = get_neib(src_nodes, adj_list, prob,num_sample = num_sample) 
        nodes_layers.insert(0, (  uniq_nodes,neib_nodes,uniq_dic))  
        src_nodes = uniq_nodes  
    assert len(nodes_layers) ==  num_layers + 1 
    return nodes_layers
 

def one_hot_neib( Y , neib,root = "./data/"):
    num_sample = Y.shape[0]
    labels = np.zeros((num_sample, 128))
    for i in range(num_sample):
        labels[i,int(Y[i])] = 1
        labels[i, neib[int(Y[i])]] = 1
    return labels

def hop_acc( labels_neib , pred):
    match = 0
    num_test = labels_neib.shape[0]
    for i in range(num_test):
        match += labels_neib[i, pred[i]]
    acc = torch.true_divide(match,num_test)
    return acc.numpy()

class Outlayer_fully( nn.Module):
    def __init__(self,  emb_size,  num_node , dropout = 0.2 ): 
        super(Outlayer_fully, self).__init__()
        self.emb_size = emb_size 
        self.dropout = dropout
        output_dim1 = int( 2* num_node)
        self.weight1 = nn.Parameter(torch.FloatTensor(emb_size* num_node , output_dim1))
        self.weight_out = nn.Parameter(torch.FloatTensor(output_dim1, num_node) )
        self.bias1 = nn.Parameter(torch.Tensor(output_dim1))
        self.bias_out = nn.Parameter(torch.Tensor(num_node))
        self.layer = nn.Sequential(nn.Linear(emb_size, num_node)) 
        self.reset_parameters() 
                                       
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_out)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.zeros_(self.bias_out)
            
    def forward(self, embs ): 
        fn1 =  torch.reshape(embs , [embs.shape[0],embs.shape[1] * embs.shape[2] ])    
        fn1 = torch.matmul(fn1, self.weight1) + self.bias1
        fn1  = F.dropout(fn1, self.dropout, training = self.training)
        pred = torch.matmul(fn1, self.weight_out) + self.bias_out
        pred  = F.dropout(pred, self.dropout, training = self.training)
        log_softmax = torch.log_softmax(pred, 1)
        return log_softmax 
    
class SelfNeibAggreg( nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0,  act = F.relu,  agg_hidden_method = 'concat'  ):  
        super(SelfNeibAggreg, self).__init__() 
        assert agg_hidden_method in ['sum', 'concat']
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.act = act 
        self.agg_hidden_method = agg_hidden_method  
        self.weight = nn.Parameter(torch.Tensor(input_dim  , hidden_dim) )
        self.reset_parameters() 
        self.dropout = dropout
        
    def reset_parameters(self): 
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, self_hidden, neib_hidden):  
        if self.agg_hidden_method == 'sum':
            hidden = self_hidden + neib_hidden
        elif self.agg_hidden_method == 'concat':
            hidden = torch.cat([self_hidden, neib_hidden], dim = 2)  
        else:
            raise ValueError('Unknown aggregate method'.format(self.agg_hidden_method))
        
        if self.act: 
            B = self_hidden.shape[0] 
            h_agg = torch.matmul(hidden, self.weight) 
            h_drop = F.dropout(h_agg, self.dropout, training=self.training) 
            return self.act(h_drop)
        else:
            h_agg = torch.matmul(hidden, self.weight)
            h_drop = F.dropout(h_agg, self.dropout, training=self.training) 
            return  h_drop 
              
class GraphSage( nn.Module): 
    def __init__(self,num_layers, input_dim, hidden_dim,  adj,prob_adj, dropout = 0, device= 'cpu' , agg_method ='mean', agg_hidden_method = 'concat'  ):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim  
        self.device = device 
        self.num_layers = num_layers  
        self.agg_method = agg_method
        self.dropout = dropout
        self.agg_hidden_method = agg_hidden_method
        self.gcn = nn.ModuleList() 
        self.dim_neib = 2 if self.agg_hidden_method == 'concat' else 1
        self.gcn.append(SelfNeibAggreg(input_dim*self.dim_neib, hidden_dim[0], dropout = self.dropout, act = F.relu, agg_hidden_method = 'concat'  ))  
        for l in range(0, len(hidden_dim) - 2):
            self.gcn.append(SelfNeibAggreg(hidden_dim[l]*self.dim_neib  , hidden_dim[l+1] ,dropout = self.dropout, act = F.relu, agg_hidden_method = agg_hidden_method    ))  
        self.gcn.append(SelfNeibAggreg(hidden_dim[-2]*self.dim_neib , hidden_dim[-1] ,  dropout = self.dropout, act = F.relu , agg_hidden_method = agg_hidden_method )  ) 
        self.outweight = nn.Parameter(torch.Tensor( hidden_dim[-1], 1))  
        self.reset_parameters() 
        self.adj = adj
        self.self_include = False
        self.prob_adj = prob_adj
        
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.outweight) 
        
        
    def forward(self, nodes_layers, raw_features ):  
        B, num_nodes, input_dim = raw_features.shape
        src_nodes  = list(range(num_nodes)) 
        nodes_layers = [(src_nodes,)]
        for i in range(self.num_layers ):
            uniq_nodes, neib_nodes, uniq_dic = self.get_neib(src_nodes, self.adj, self.prob_adj )
            nodes_layers.insert(0, (  uniq_nodes,neib_nodes,uniq_dic))  
            src_nodes = uniq_nodes
            
        assert len(nodes_layers) == self.num_layers + 1 
        
        nodes_features = raw_features
        for index in range(1, self.num_layers +1): 
            self_nodes = nodes_layers[index][0]
            neib_info = nodes_layers[index - 1]
            neib_features = self.aggregator(  self_nodes , nodes_features  ,neib_info)
            sage_layer = self.gcn[index - 1]
            if index > 1:
                self_nodes = self.node_map(self_nodes,   neib_info)
            emb_cur = sage_layer(nodes_features[:, self_nodes,:], neib_features )
            nodes_features = emb_cur  
        return nodes_features 
    
    def get_neib( self, nodes, adj, prob,num_sample = 7):  
        _set = set 
        to_neibs = [ self.adj[int(node)] for node in nodes] 
        prob_node =  [ prob[[int(node)]*len(self.adj[int(node)]), self.adj[int(node)]] for node in nodes]  
        if not num_sample is None:
            sample_neib = [set(np.random.choice(  to_neibs[ind_neib] , num_sample, p =  (prob_node[ind_neib]), replace = False )) if len( to_neibs[ind_neib] ) >= num_sample else  set(to_neibs[ind_neib]) for ind_neib in range(len(to_neibs)) ]
        else:
            sample_neib = to_neibs   
        sample_neib = [sample | set([nodes[i]]) for i , sample in enumerate(sample_neib)   ] 
        unique_nodes_list = list(set.union(*sample_neib))  
        i = list(range(len(unique_nodes_list))) 
        unique_dic = dict(list(zip(unique_nodes_list, i))) 
        return unique_nodes_list, sample_neib, unique_dic
 
    
    def node_map(self, self_nodes,   neib_info):
        uniq_nodes, neib_nodes, uniq_dic = neib_info
        assert len(neib_nodes) == len(self_nodes)  
        index = [uniq_dic[x] for x in self_nodes]
        return index 
    
    def aggregator(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs 
        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.self_include:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))] 
        samp_neighs_weights = [ self.prob_adj[nodes[i], list(samp_neighs[i])   ]  for i in range(len(samp_neighs)) ]
            
        if  (pre_hidden_embs.shape[1]) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[:,torch.LongTensor(unique_nodes_list),:] 
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))] 
        samp_neighs_weights = [item for sublist in samp_neighs_weights for item in sublist] 
        mask[row_indices, column_indices] = torch.Tensor(samp_neighs_weights) 
            
        if self.agg_method == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            mask_batch= mask.repeat(embed_matrix.shape[0], 1,1)
            aggregate_feats = torch.matmul(mask , embed_matrix ) 
        elif self.agg_method == 'MAX': 
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = [] 
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)
 
                
        return aggregate_feats 
    
def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma_list = []
    for i in range(len(dist)):
        pos = 0
        while dist[i, pos + 1] != np.inf:
            pos += 1
            if pos >= len(dist[0]) - 1:
                break
        sigma_list.append(dist[i, pos])
    sigma2 = np.mean(sigma_list)**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W 
    
    
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight) 
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)            

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, bias = True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid[0])
        self.gc2 = GraphConvolution(nhid[0], nhid[1]) 
        self.outweight = nn.Parameter(torch.FloatTensor(  nhid[1],  nclass)) 
        self.dropout = dropout
        self.bias = bias
        if bias: 
            self.outbias = nn.Parameter(torch.FloatTensor(nclass)) 
        self.reset_parameters()

    def reset_parameters(self): 
        torch.nn.init.xavier_uniform_(self.outweight) 
        if self.bias is not None:  
            torch.nn.init.zeros_(self.outbias)



    def forward(self, x, adj):
        N, M  = x.shape   
        h1 = F.relu(self.gc1(x, adj)) 
        h1 = F.dropout(h1, self.dropout, training = self.training) 
        h2 = F.relu(self.gc2(h1, adj)) 
        h2 = F.dropout(h2, self.dropout, training=self.training)  
        out = torch.matmul(h2, self.outweight)    
        if self.bias is not None:
            out += self.outbias 
        return F.log_softmax(out, dim=1)

def train( epoch, model,optimizer,  features, labels, adj, ind_label, fastmode, id_val , batch_size):
    t = time.time()
    model.train()
    optimizer.zero_grad()  
    output = model(features , adj) 
    loss_train = F.nll_loss(output[ind_label] , labels[ind_label])
    acc_train =accuracy(output[ind_label] , labels[ind_label] ) 
    loss_train.backward()
    optimizer.step()

    if not fastmode: 
        model.eval()
        output = model(features , adj)

    output = model(features , adj) 
    loss_val = F.nll_loss(output[id_val]  , labels[id_val])
    acc_val = accuracy(output[id_val] , labels[id_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_train


def test(model,features,labels, neib,  adj, id_test   ):
    model.eval()
    output = model(features , adj)  
    loss_test = F.nll_loss(output[id_test] , labels[id_test])
    acc_test = accuracy(output[id_test] , labels[id_test])
    multi_labels = torch.LongTensor(one_hot_neib(labels[id_test] , neib))
    match = 0
    pred = output[id_test].max(1)[1] 
    for i in range(len(id_test)):
        match += multi_labels[i, pred[i]]  
    acc_hop = torch.true_divide(match,len(id_test))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
         "1-hop accuracy = {:.4f}".format(acc_hop))
    return acc_test, acc_hop 

def accuracy(  output, labels ): 
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) 
 
def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx
 
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    D_inv_half = np.power(rowsum, -0.5).flatten()
    D_inv_half[np.isinf(D_inv_half)] = 0.
    D_mat = sp.diags(D_inv_half)
    return adj.dot(D_mat).transpose().dot(D_mat).tocoo()

def preprocess_adj(adj):
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj#sparse_to_tuple(adj)

def Wz(logit, neib, k_neib = 30  ):
    logit = torch.reshape(logit, [logit.shape[0], logit.shape[1]])
    pred = logit.max(1)[1]
    with torch.no_grad():
        l = pred.numpy()
        Z_loc = torch.zeros(logit.shape)
        for i in range(Z_loc.shape[0]):
            i_neib = list(neib[l[i]])
            i_neib.append(l[i])
            Z_loc[i, i_neib] = logit[i, i_neib]
        Z = Z_loc.numpy() 
        dz_cos, idz_cos = distance_sklearn_metrics(Z , k=min(k_neib, Z.shape[0] ), metric='cosine') 
        dz_2, idz_2 = distance_sklearn_metrics(Z , k=min(k_neib, Z.shape[0]), metric='euclidean')# 

        Wz_cos =  adjacency(dz_cos, idz_cos)
        Wz_2 =  adjacency(dz_2, idz_2)

        Wz_cos_n = preprocess_adj(Wz_cos)
        Wz_2_n = preprocess_adj(Wz_2)
        return Wz_cos, Wz_2, Wz_cos_n, Wz_2_n

def constructW_stageI(embs_best, logits_best ,neib,  k_neib = 30  ):
    Wz_cos, Wz_2, Wz_cos_n, Wz_2_n    = Wz(logits_best,neib,  k_neib)
    A = Wz_cos_n
    Adj = torch.sparse.FloatTensor(torch.stack((torch.LongTensor(A.row),torch.LongTensor( A.col))), torch.FloatTensor(A.data)  )
    dics_w = {}
    dics_w['Adj'] = Adj;  
    return Adj 

def map_label():
    sets = [[84, 85], [22, 24], [28, 29, 83], [20, 31], [33, 124], [19, 26], [40, 42], [116, 127]]
    y_map = get_y_map(128, sets)
    map_inv = {}
    for k,v in y_map.items():
        if v in map_inv:
            map_inv[v].append(k)
        else:
            map_inv[v] = [k] 
    return map_inv, y_map

def A_labels(A):
    A_119 = {}
    map_inv, y_map = map_label()
    for i in range(119):
        bus = map_inv[i]
        A_119[i] = []
        for b in bus: 
            for l in A[b]:
                if y_map[l] not in A_119[i]:
                    A_119[i].append(y_map[l])
    return A_119
  
