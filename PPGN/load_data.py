import os, scipy
import torch   
import scipy.sparse as sp 
import numpy as np 
import pickle

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

 

def load_allV_data(name_train = 'train_set_allV.npz',name_test = 'test_set_allV.npz', root = "./data/" ): 
    train_dic = np.load(os.path.join(root, name_train))
    test_dic = np.load(os.path.join(root, name_test))
    all_data,  labels = np.concatenate((train_dic['arr_0'][:, :, :6  ], test_dic['arr_0'][:, :, :6  ]), 0), np.concatenate((train_dic['arr_1'], test_dic['arr_1']))
    features = all_data[:, :,   :6]  
    return features, labels 

def load_data_single_observ( name_train, name_test, num_labelper ,seed, measured_index,    phase =  'pp',  device = 'cpu', random = False ,  root = "./data/"):
    assert phase in [ 'sp3' ,   'ppg' , 'pp' ,'ground','ind_single'  ]
    with open(os.path.join(root, 'types_sp3.pickle'), 'rb') as f: 
        dic_types= pickle.load(f) 
    with open(os.path.join(root, 'neib.pickle'), 'rb') as f:
        dic = pickle.load(f)
    neib_orig = dic['neib']
    features_orig, labels = load_allV_data(name_train, name_test)
    ind_single = dic_types[phase]
    features = np.zeros_like(features_orig) 
    ind_partial = measured_index 
    features[:,  ind_partial ,:]  = features_orig[:,  ind_partial , :]   
    neib_observ = {}
    for i in range(len(measured_index)): 
        neib_observ[i] = neib_orig[ ind_partial[i] ]
    sets = [[84, 85], [22, 24], [28, 29, 83], [20, 31], [33, 124], [19, 26], [40, 42], [116, 127]]
    y_map = get_y_map(128, sets)
    y_map_inv = dict((v, k) for k, v in y_map.items()) 
    for i in range(np.shape(labels)[0]):
        labels[i] = y_map[labels[i]] 
    features, labels = torch.FloatTensor(features[ind_single] ).to(device), torch.LongTensor(labels[ind_single] ).to(device)
    ind_train,  ind_test = list(range(int( features.shape[0]/2))), list(range(int(features.shape[0]/2), int(features.shape[0]  )) ) 
    ind_labels =  randomize_labels(seed,labels  , int(num_labelper) ) 
    ind_test = [item for item in list(range(labels.shape[0])) if item not in ind_labels]
    ind_measured = measured_index  
    return  features, labels, ind_train ,  ind_test,    neib_observ,ind_labels, ind_measured


def load_all_types(name_train, name_test,num_labelper, measured_index, seed = 842, root = "./data/"):
    features_sp3, labels_sp3, _,_,    neib_observ,ind_labels_sp3, ind_measured= load_data_single_observ( name_train, name_test, num_labelper , seed, measured_index, phase = 'sp3'    ) 
    features_pp, labels_pp, _,_,    neib_observ,ind_labels_pp, ind_measured= load_data_single_observ( name_train, name_test, num_labelper , seed, measured_index, phase = 'pp' ,  )   
    features_ppg, labels_ppg, _,_,    neib_observ,ind_labels_ppg, ind_measured= load_data_single_observ( name_train, name_test,  num_labelper ,seed, measured_index, phase = 'ppg'    )   
    features = torch.cat((features_sp3, features_pp, features_ppg), dim = 0)
    labels = torch.cat((labels_sp3, labels_pp, labels_ppg), dim = 0) 
    ind_labels_pp1 = [item +int( labels_sp3.shape[0]) for item in ind_labels_pp] 
    ind_labels_ppg1 = [item +int(labels_sp3.shape[0]) +int(labels_pp.shape[0] ) for item in ind_labels_ppg] 
    ind_labels =  list(ind_labels_sp3) + ind_labels_pp1+ind_labels_ppg1  
    ind_train, ind_val,ind_test = list(range(int( features.shape[0]/2))),list(range(int( features.shape[0]/2))), list(range(int(features.shape[0]/2), int(features.shape[0]  )) ) 
    return features, labels, ind_train ,  ind_test,    neib_observ,ind_labels, ind_measured
 
class dataCenter(object):
        def __init__(self, name_train, name_test,      num_labelper, seed, measured_index, faulttype , dataSet = 'loc', root = "./data/"):
            super( dataCenter, self).__init__() 
            if faulttype != 'all':
                features, labels, ind_train ,  ind_test,    neib_observ,ind_labels, ind_measured= load_data_single_observ( name_train, name_test, num_labelper ,seed, measured_index,  phase = faulttype )
            else:
                features, labels, ind_train ,  ind_test,    neib_observ,ind_labels, ind_measured= load_all_types(name_train, name_test, num_labelper , measured_index, seed = seed )
            setattr(self, dataSet+'_test', np.array(ind_test))
            setattr(self, dataSet+'_val', np.array(ind_train))
            setattr(self, dataSet+'_train', np.array(ind_labels))

            setattr(self, dataSet+'_feats', features)
            setattr(self, dataSet+'_labels', labels) 
            setattr(self, dataSet+'_adj_lists', neib_observ )  
            setattr(self, dataSet+'_bus_measured', ind_measured)  
            
def create_label_permutations(seed,labels,T,m,multiplier=None):
    np.random.seed(seed) 
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
 
def randomize_labels(seed, L,m):

    perm = create_label_permutations(seed,L,1,[m])

    return perm[0]    
 
def adjacency(dist, idx): 
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

def A_dist(k =3, root =  "./data/"):
    dist_matrix = np.load(os.path.join(root,'dist_matrix.npy'))
    dist_graph = np.load(os.path.join(root,'dist_graph.npy')) 
    START = 1 
    idx = np.argsort(dist_matrix)[:, START:k+1]  
    dist_matrix.sort()
    dist_matrix = dist_matrix[:, START:k+1]
    A_dis =  adjacency(dist_matrix, idx).astype(np.float32) 
    return A_dis 
 
def select_A_prob(k, name = 'A_adm',sparse = False, root = "./data/"): # 'A_len', 'A_short', 'A_adm'
    with open(os.path.join(root, 'A_normalized_rowsum.pickle'), 'rb') as f:
        dics = pickle.load(f)
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
 