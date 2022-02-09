import os, scipy
import torch
import scipy.sparse as sp
import numpy as np
import pickle
import torch

global seed
seed = 842

def load_checkpoint(models , optimizer , savebest): 
    start_epoch = 0
    if os.path.isfile(savebest):
        print("=> loading checkpoint '{}'".format(savebest))
        checkpoint = torch.load(savebest)
        start_epoch = checkpoint['epoch'] 
        models[0].load_state_dict(checkpoint['state_dict'][0])
        models[1].load_state_dict(checkpoint['state_dict'][1])
        optimizer.load_state_dict(checkpoint['optimizer']) 
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(savebest, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(savebest))

    return models , optimizer, start_epoch 


def load_allV_data(name_train = 'train_set_allV_37nodes_1pu.npz',name_test = 'test_set_allV_37nodes_1pu.npz', root = "./data/" ):# _1.4_times_loads_vary.npz allV has 128 measurements
    train_dic = np.load(os.path.join(root, name_train))
    test_dic = np.load(os.path.join(root, name_test))
    all_data,  labels = np.concatenate((train_dic['arr_0'][:, :, :6  ], test_dic['arr_0'][:, :, :6  ]), 0), np.concatenate((train_dic['arr_1'], test_dic['arr_1']))
    features = all_data[:, :,   :6]
    types =np.r_[train_dic['arr_2'] ,test_dic['arr_2'] ]
    dic_types = {}
    dic_types['sp3'] = list(np.where( types == 'sp3')[0])
    dic_types['pp'] = list(np.where( types  == 'pp')[0])
    dic_types['ppg'] = list(np.where(np.array(types) == 'ppg')[0])
    return features, labels, dic_types

 

def load_data_single_observ(  num_labelper ,measured_index, seed = 842,  name_train = 'train_set_allV_37nodes_1pu.npz',name_test = 'test_set_allV_37nodes_1pu.npz',  phase =  'ppg',  device = 'cpu', random = False ,  root = "./data/"):
    assert phase in [  'sp3' , 'ppg' , 'pp'   ]
    features_orig, labels, dic_types = load_allV_data(name_test = name_test)
    ind_single = dic_types[phase]
    features = np.zeros_like(features_orig)
    ind_partial = measured_index
    features[:,  ind_partial ,:]  = features_orig[:,  ind_partial , :]
    features, labels = torch.FloatTensor(features[ind_single] ).to(device), torch.LongTensor(labels[ind_single] ).to(device) 
    ind_train = list(range(int( features.shape[0] )))
    ind_labels =  randomize_labels(labels[ind_train]  , int(num_labelper) )
    ind_test = [item for item in range(features.shape[0]) if item not in ind_labels]
    ind_measured = measured_index
    return  features, labels,     ind_labels


def load_all_types(num_labelper, measured_index, seed = 842, name_train = 'train_set_allV_37nodes_1pu.npz',name_test = 'test_set_allV_37nodes_1pu.npz'):
    features_sp3, labels_sp3,  ind_labels_sp3 = load_data_single_observ( num_labelper , measured_index,phase = 'sp3', seed = seed  )#features = torch.FloatTensor(features).to(device)
    features_pp,  labels_pp,  ind_labels_pp = load_data_single_observ( num_labelper , measured_index,phase = 'pp' , seed = seed  )
    features_ppg, labels_ppg,  ind_labels_ppg = load_data_single_observ( num_labelper ,measured_index, phase = 'ppg', seed = seed   )
    features = torch.cat((features_sp3, features_pp, features_ppg), dim = 0)
    labels = torch.cat((labels_sp3, labels_pp, labels_ppg), dim = 0)
    ind_labels_pp1 = [item +int( labels_sp3.shape[0]) for item in ind_labels_pp]
    ind_labels_ppg1 = [item +int(labels_sp3.shape[0]) +int(labels_pp.shape[0] ) for item in ind_labels_ppg]
    ind_labels =  list(ind_labels_sp3) + ind_labels_pp1+ind_labels_ppg1
    ind_test = [item for item in range(features.shape[0]) if item not in ind_labels]
    return features, labels, ind_labels ,  ind_test,  measured_index

class dataCenter(object):
        def __init__(self,      num_labelper,measured_index, seed = 842,   dataSet = 'loc', name_train = 'train_set_allV_37nodes_1pu.npz',name_test = 'test_set_allV_37nodes_1pu.npz'):
            super( dataCenter, self).__init__()
            #features, labels, ind_train ,  ind_test,    neib_observ,ind_labels, ind_measured= load_data_single_observ( num_labelper = num_labelper, phase = phase )
            features, labels, ind_train ,  ind_test,     ind_measured= load_all_types( num_labelper ,measured_index,  seed = seed )
            setattr(self, dataSet+'_test', np.array(ind_test))
            setattr(self, dataSet+'_val', np.array(ind_train))
            setattr(self, dataSet+'_train', np.array(ind_train))

            setattr(self, dataSet+'_feats', features)
            setattr(self, dataSet+'_labels', labels)
            setattr(self, dataSet+'_bus_measured', ind_measured)
            
            
            
 
            
def create_label_permutations(labels,T,m,multiplier=None):
    np.random.seed(seed)
    #Find all unique labels >= 0
    #Negative numbers indicate unlabeled nodes
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels>=0]

    perm = list()
    n = labels.shape[0]
    J = np.arange(n).astype(int)
    for k in range(T):
        for i in m:
            L = []
            ind = 0
            for l in unique_labels:
                I = labels==l
                K = J[I]
                if multiplier is None:
                    L = L + np.random.choice(K,size=i,replace=False).tolist()
                else:
                    sze = int(np.round(i*multiplier[ind]))
                    L = L + np.random.choice(K,size=sze,replace=False).tolist()
                ind = ind + 1
            L = np.array(L)
            perm.append(L)

    return perm
 
def randomize_labels(L,m):

    perm = create_label_permutations(L,1,[m])

    return perm[0]
 
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

def A_dist(k =3, root =  "/Users/wenting/Documents/04_research/02_Graph_learning/01_simu/02_dataGenerator/code_GCN/data/"):
    # k= 3 within k hops shortest distances
    dist_matrix = np.load(os.path.join(root,'dist_matrix_37nodes.npy'))
    START = 1
    idx = np.argsort(dist_matrix)[:, START:k+1] # not include itself
    dist_matrix.sort()
    dist_matrix = dist_matrix[:, START:k+1]
    A_dis =  adjacency(dist_matrix, idx).astype(np.float32)
    return A_dis#, dist_matrix, dist_graph
    
def select_A_prob(k, name = 'A_short',sparse = False, root = "./data/"): # 'A_len', 'A_short', 'A_adm'
    if name == 'A_short':
        A= A_dist(k = k)
    else:
        A = dics[name] 
    Adj = A.todense()
    prob_A = np.asarray(Adj).astype('float64')
    for test in range(prob_A.shape[0]):
        prob_A[test,:] =  (prob_A[test,:]/ (prob_A[test,:]).sum())
    A = {}
    for node in range(prob_A.shape[0]):
        A[node] =   np.array(np.where(prob_A[node, :] > 0)).flatten()
    return A, prob_A



