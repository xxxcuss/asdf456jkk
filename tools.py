import pdb
import copy
import torch
import warnings
import numpy as np
from scipy import stats
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

# Federated averaging: FedAgg
def FedAggServer(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAggClient(args, w, error, w_glob): #error is a parameter tensor
    if args.client_agg == 'avg':
        w_out = copy.deepcopy(w[0])
        for k in w_out.keys():
            for i in range(1, len(w)):
                w_out[k] += w[i][k]
            w_out[k] = torch.div(w_out[k], len(w))

    elif args.client_agg == 'trim_mean':
        w_out = trim_mean(args, w)

    elif args.client_agg == 'median':
        w_out = copy.deepcopy(w[0])
        for k in w_out.keys():
            temp = torch.zeros(len(w), w[0][k].view(-1).size(0))
            for i in range(1, len(w)):
                temp[i] = w[i][k].view(-1)
            w_out[k] = torch.median(temp, dim=0)[0].reshape(w[0][k].size())
    
    elif args.client_agg == 'sparsefed':
        #clip & average
        w_glob_t = paradict2tensor(error, w_glob)
        grad_all_t = torch.zeros((len(w), error.size(0)))
        for i in range(len(grad_all_t)):
            grad_all_t[i] = (paradict2tensor(error, w[i]) - w_glob_t) #gradients
         
        minc = torch.clamp(args.L / torch.abs(torch.norm(grad_all_t, dim=1)), max=1)
        w_all_t = grad_all_t * minc.unsqueeze(1).repeat(1, grad_all_t.size(1))
        w_avg = torch.mean(w_all_t, dim=0)

        #topk & update
        W_t = w_avg + error
        delta_t = topk(W_t, args.k)
        error = W_t - delta_t

        w_out_ = w_glob_t + delta_t
        w_out = paratensor2dict(w[0], w_out_)

    elif args.client_agg == 'krum':
        w_glob_t = paradict2tensor(error, w_glob)
        grads = torch.zeros((len(w), error.size(0)))
        for i in range(len(grads)):
            grads[i] = paradict2tensor(error, w[i]) - w_glob_t #gradients

        n_workers = len(grads)
        distances = {i: {j: None for j in range(n_workers) if i != j} for i in range(n_workers)}
        closest_sums = torch.zeros(n_workers).cuda()

        for idx, g in enumerate(grads):
            for jdx, j in enumerate(grads):
                if idx != jdx:
                    if distances[jdx][idx] is not None:
                        distances[idx][jdx] = distances[jdx][idx]
                    else:
                        distances[idx][jdx] = (g - j).norm(p=2)
            dist_array = torch.tensor([val for key, val in distances[idx].items()]).cuda()
            dist_array = torch.sort(dist_array)[0]
            closest_dists = dist_array[:-3]
            closest_sums[idx] = closest_dists.sum()

        delta_t = grads[torch.sort(closest_sums)[1][0]]

        w_out_ = w_glob_t + delta_t
        w_out = paratensor2dict(w[0], w_out_)

    elif args.client_agg == 'bulyan':
        f = 1
        w_glob_t = paradict2tensor(error, w_glob)
        grads = torch.zeros((len(w), error.size(0)))
        for i in range(len(grads)):
            grads[i] = paradict2tensor(error, w[i])

        n_workers = len(grads)
        theta = len(grads) - 2 * f
        s = []
        # compute distances between all models
        distances = {i: {j: None for j in range(n_workers) if i != j} for i in range(n_workers)}
        for idx, g in enumerate(grads):
            for jdx, j in enumerate(grads):
                if idx != jdx:
                    if distances[jdx][idx] is not None:
                        distances[idx][jdx] = distances[jdx][idx]
                    else:
                        distances[idx][jdx] = (g - j).norm(p=2)

        while len(s) < theta:
            # get new candidate based on the output of krum
            model_idx = bulyan_krum(distances, n_workers)
            # remove candidate from distances for recursion
            distances = {key_outer: {key_inner: val_inner for key_inner, val_inner in val_outer.items() if key_inner != model_idx} for key_outer, val_outer in distances.items() if key_outer != model_idx}
            # add candidate to s
            grad = grads[model_idx].cpu()
            s.append(grad)
        # return the trimmed mean of the candidate set
        #return torch.stack(s).sort()[0][f:-f].mean()
        w_out = paratensor2dict(w[0], torch.stack(s).mean(0))

    return w_out, error

def bulyan_krum(distances, n_workers):
    # keep an array of the sum of the distances of all other models except for the 3 furthest to this worker
    closest_sums = torch.ones(n_workers) * float('inf')
    for idx, dists in distances.items():
        dist_array = torch.tensor([val for key, val in dists.items()])
        dist_array = torch.sort(dist_array)[0]
        closest_dists = dist_array[:-(1+2)]
        closest_sums[idx] = closest_dists.sum()
    # return the model that is "overall closer" to all the other models"
    argmin_closest = torch.sort(closest_sums)[1][0]

    return argmin_closest

def trim_mean(args, w):
    w_out = copy.deepcopy(w[0])
    for k in w_out.keys():
        temp = torch.zeros(len(w), w[0][k].view(-1).size(0))
        for i in range(1, len(w)):
            temp[i] = w[i][k].reshape(-1)
        w_out[k] = torch.tensor(stats.trim_mean(temp, 1.0/args.num_users+0.01, axis=0).reshape(w[0][k].size()))
        pdb.set_trace()
    return w_out

def paratensor2dict(temp, t):
    index = 0
    dict_list = []
    
    for k in temp.keys():
        length = len(temp[k].view(-1))
        tensor = t[index:index+length].reshape(temp[k].size())
        dict_list.append((k, tensor))
        index += length

    d = OrderedDict(dict_list)
    return d

def paradict2tensor(temp, d): #sum(v.view(-1).size(0) for k, v in d.items())
    index = 0
    t = torch.zeros(len(temp))

    for k in d.keys():
        tensor = d[k].reshape(-1)
        length = len(tensor)
        t[index:index+length] = tensor
        index += length

    return t

def topk(W_t, k):
    delta_t = torch.zeros(len(W_t))
    top_k = torch.topk(W_t**2, k)
    for index in range(len(top_k[0])):
        delta_t[top_k[1][index]] = W_t[top_k[1][index]]
    return delta_t

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

def calculate_matrix(cnf_matrix):

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP0 = FP.astype(float)[0] #class 0
    FN0 = FN.astype(float)[0]
    TP0 = TP.astype(float)[0]
    TN0 = TN.astype(float)[0]

    FP1 = FP.astype(float)[1] #class 1
    FN1 = FN.astype(float)[1]
    TP1 = TP.astype(float)[1]
    TN1 = TN.astype(float)[1]

    rates0, rates1 = np.zeros(4), np.zeros(4)
    # Sensitivity, hit rate, recall, or true positive rate (TPR)
    warnings.filterwarnings('error')
    try:
        rates0[0], rates1[0] = cnf_matrix.sum(axis=1)[0], cnf_matrix.sum(axis=1)[1]
        #TP0/(TP0+FN0), TP1/(TP1+FN1)
    except RuntimeWarning:
        pdb.set_trace()
    # Specificity or true negative rate (TNR)
    rates0[1], rates1[1] = cnf_matrix[0,0], cnf_matrix[1,0]
    #TN0/(TN0+FP0), TN1/(TN1+FP1)
    # Fall out or false positive rate (FPR)
    rates0[2], rates1[2] = cnf_matrix[0,1], cnf_matrix[1,1]
    #FP0/(FP0+TN0), FP1/(FP1+TN1)
    # False negative rate (FNR)
    rates0[3], rates1[3] = rates0[2] / rates0[0], rates1[2] / rates1[0]
    #FN0/(TP0+FN0), FN1/(TP1+FN1)
    
    return rates0, rates1

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))   