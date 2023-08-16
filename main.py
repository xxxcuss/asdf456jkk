#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch.utils.data import DataLoader, Dataset
import os.path

from pandas import DataFrame
from tools import *
from model import *
from dataloader import *

import random
import numpy as np
import os
import pdb
import opts
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torch.autograd import grad, Variable

def main(args):
    SEED, num_users, epochs, frac, lr = args.seed, args.num_users, args.epochs, args.frac, args.lr

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))    

    #===================================================================
    #program = "SFLV1 ResNet18 on HAM10000"
    #print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #------------ Training And Testing  -----------------
    net_glob_client.train()
    #copy weights
    w_glob_client = net_glob_client.state_dict()

    dataset_train, dataset_test, dict_users, dict_users_test = data_load(args)
    
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    attacker_id = args.attacker_id.split(',')
    delta_data, delta_smash = {}, {}
    for aid in attacker_id:
        delta_data[int(aid)] = torch.zeros(len(dict_users[int(aid)]),args.chan,64,64).to(device)
        delta_smash[int(aid)] = torch.zeros(len(dict_users[int(aid)]),64,16,16).to(device) #256x64x16x16

    error = torch.zeros(sum(v.view(-1).size(0) for k, v in w_glob_client.items()))
    acc_list = []
    for iter in range(epochs):
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace = False)
        w_locals_client = []
        
        if iter < args.attack_before or iter > args.attack_after-1:
            args.attack = True
        if iter > args.attack_before-1 and iter < args.attack_after:
            args.attack = False

        for idx in idxs_users:
            local = Client(args, net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # Training ------------------

            w_client, delta_data, delta_smash = local.train(copy.deepcopy(net_glob_client).to(device), delta_data, delta_smash)
            w_locals_client.append(copy.deepcopy(w_client))
            # Testing -------------------
            last_acc = local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter) ###
        
        acc_list.append(last_acc)
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
        print("-----------------------------------------------------------")
        print("------ FedServer: Federation process at Client-Side ------- ")
        print("-----------------------------------------------------------")
        w_glob_client, error = FedAggClient(args, w_locals_client, error, w_glob_client)   
        
        # Update client-side global model 
        net_glob_client.load_state_dict(w_glob_client)    
    
    print('All accuracy: ', acc_list)
    print("Training and Evaluation completed!")
    return 
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Server-side function associated with Training 
def train_server(args, device, fx_client, y, l_epoch_count, l_epoch, idx, len_batch, last_iter):
    # For Server Side Loss and Accuracy
    #attacker_id = args.attacker_id.split(',')
    #if str(idx) in attacker_id and args.attack:
    #    lr = args.lr #alr
    #else:
    lr = args.lr
    
    global net_model_server, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user
    
    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
    
    # train and update
    optimizer_server.zero_grad()
    fx_client, y = fx_client.to(device), y.to(device)

    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()

    if last_iter == False:
        pass
        #prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc, loss))
    else:
        optimizer_server.step()
    
        batch_loss_train.append(loss.item())
        batch_acc_train.append(acc.item())

        # Update the server-side model for the current batch
        net_model_server[idx] = copy.deepcopy(net_server)
    
        # count1: to track the completion of the local batch associated with one client
        count1 += 1

    if count1 == len_batch and last_iter == True:

        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        if len(idx_collect) == args.num_users:
            fed_check = True                  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side----------- output print and update is done in evaluate_server()
            w_glob_server = FedAggServer(w_locals_server)   
            
            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)    
            net_model_server = [net_glob_server for i in range(args.num_users)]
            
            w_locals_server = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):

    global net_model_server, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server 
    global loss_test_collect, acc_test_collect, count2, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    global batch_cnf, cnf_test_collect_user, cnf_test_collect

    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        preds = fx_server.max(1, keepdim=True)[1].squeeze(1)
        cnf = confusion_matrix(y.cpu(), preds.cpu(), labels=np.array(range(args.classes)))

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        batch_cnf += cnf
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            cnf_test = batch_cnf
            
            batch_acc_test = []
            batch_loss_test = []
            batch_cnf = np.zeros((args.classes, args.classes))
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                cnf_test_all = cnf_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                cnf_test_collect_user += cnf_test_all
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
                cnf_test_all_user = cnf_test_collect_user

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                cnf_test_collect += cnf_test_all_user
                acc_test_collect_user = []
                loss_test_collect_user= []
                cnf_test_collect_user = np.zeros((args.classes, args.classes))

                cnf_matrix = calculate_matrix(cnf_test_all_user)
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print(' Class 0: SUM {:.3f}, Pre0 {:.3f}, Pre1 {:.3f}, Rate {:.3f}, '.format(cnf_matrix[0][0],cnf_matrix[0][1],cnf_matrix[0][2],cnf_matrix[0][3]))
                print(' Class 1: SUM {:.3f}, Pre0 {:.3f}, Pre1 {:.3f}, Rate {:.3f}, '.format(cnf_matrix[1][0],cnf_matrix[1][1],cnf_matrix[1][2],cnf_matrix[1][3]))
                print("==========================================================")
                
                return round(float(acc_avg_all_user), 4)

    return 0

#==============================================================================================================
#                                       Clients-side Program 
#==============================================================================================================
# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, args, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.args = args
        self.idx = idx
        self.device = device
        self.attacker_id = args.attacker_id.split(',')
        
        #self.selected_clients = []
        if str(self.idx) in self.attacker_id and args.attack:
            self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = args.attack_batch_size, shuffle = False)
            self.lr = args.alr
            self.local_ep = args.local_epoch
            self.batch_size = args.attack_batch_size
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = args.batch_size, shuffle = True)
            self.lr = lr
            self.local_ep = 1
            self.batch_size = args.batch_size

        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = args.batch_size, shuffle = True)
        
    def train(self, net, delta_data, delta_smash):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)

            for batch_idx, (images, labels) in enumerate(self.ldr_train): #images: 256x3x64x64 for HAM
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()

                if str(self.idx) in self.attacker_id and args.attack:
                    if len(images) == self.batch_size:
                        dd = delta_data[self.idx][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] 
                        ds = delta_smash[self.idx][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                    else:
                        dd = delta_data[self.idx][batch_idx*self.batch_size:]
                        ds = delta_smash[self.idx][batch_idx*self.batch_size:]
                    
                    if self.args.dataset_attack or self.args.smashed_attack: ###maybe for weight attack
                    	iter_num = self.args.iter_num
                    else:
                    	iter_num = 1

                    for num in range(iter_num):
                        images = images + self.args.iur * dd
                        images = Variable(images, requires_grad=True)
                        #---------forward prop-------------
                        with torch.backends.cudnn.flags(enabled=False):
                            fx = net(images)
                        
                        fx = fx + self.args.iur * ds ###
                        client_fx = fx.clone().detach().requires_grad_(True) #smashed data

                        if self.args.label_attack:
                        	labels = (labels + 1) % self.args.classes
                            #labels += (labels == 0).int()
                        # Sending activations to server and receiving gradients from server
                        if num == iter_num-1:
                            dfx = train_server(self.args, self.device, client_fx, labels, iter, self.local_ep, self.idx, len_batch, True)
                        else:
                            dfx = train_server(self.args, self.device, client_fx, labels, iter, self.local_ep, self.idx, len_batch, False)
                        #--------backward prop -------------
                        fx.backward(dfx) #gradients

                        if self.args.dataset_attack:
                            dd += images.grad.data
                        if self.args.smashed_attack:
                            ds += client_fx.grad.data
                        
                        if self.args.weight_attack:
                            temp_net = net.state_dict()
                            for k, v in net.named_parameters():
                                temp_net[k] += self.lr * v.grad.data
                                v.grad.data.zero_()
                            net.load_state_dict(temp_net)
                        else:
                            optimizer_client.step()
                            #pass
                    
                    #if batch_idx == len(self.ldr_train) -1 and iter % 5 == 0: 
                    #    print(fx[0,0,0])
                    if self.args.dataset_attack:
                        if len(images) == self.batch_size: 
                            delta_data[self.idx][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] = dd
                        else:
                            delta_data[self.idx][batch_idx*self.batch_size:] = dd
                    if self.args.smashed_attack:
                        if len(images) == self.batch_size: 
                            delta_smash[self.idx][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] = ds
                        else:
                            delta_smash[self.idx][batch_idx*self.batch_size:] = ds

                else:    
                    fx = net(images)
                    client_fx = fx.clone().detach().requires_grad_(True) #smashed data
                
                    # Sending activations to server and receiving gradients from server
                    dfx = train_server(self.args, self.device, client_fx, labels, iter, self.local_ep, self.idx, len_batch, True)
                
                    #--------backward prop -------------
                    fx.backward(dfx) #gradients

                    optimizer_client.step()

            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict(), delta_data, delta_smash 
    
    def evaluate(self, net, ell):
        net.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                acc = evaluate_server(fx, labels, self.idx, len_batch, ell) 
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
        return acc 

'''
#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)     
'''
#=============================================================================
#                         Program Completed
#=============================================================================

if __name__ == '__main__':
    
    # to print train - test together in each round-- these are made global
    starttime = datetime.datetime.now()
    args = opts.parse_opt()
    acc_avg_all_user_train = 0
    loss_avg_all_user_train = 0
    count1 = 0
    count2 = 0

    loss_train_collect = []
    acc_train_collect = []
    batch_acc_train = []
    batch_loss_train = []
    loss_train_collect_user = []
    acc_train_collect_user = []
    idx_collect = []
    w_locals_server = []
    
    loss_test_collect = []
    acc_test_collect = []

    batch_acc_test = []
    batch_loss_test = []

    loss_test_collect_user = []
    acc_test_collect_user = []

    batch_cnf = np.zeros((args.classes, args.classes))
    cnf_test_collect_user = np.zeros((args.classes, args.classes))
    cnf_test_collect = np.zeros((args.classes, args.classes))
    #client idx collector
    l_epoch_check = False
    fed_check = False
    # Initialization of net_model_server and net_server (server-side model)
    #===================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_glob_client = ResNet18_client_side(args, Baseblock)
    net_glob_server = ResNet18_server_side(args, Baseblock, [2,2,2], args.classes) #7 is my numbr of classes

    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)    
        net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

    net_glob_client.to(device)
    net_glob_server.to(device)    

    net_model_server = [net_glob_server for i in range(args.num_users)]
    net_server = copy.deepcopy(net_model_server[0]).to(device)
    
    w_glob_server = net_glob_server.state_dict()

    print(args)
    print('Model parameters: ',sum(v.view(-1).size(0) for k, v in net_glob_client.state_dict().items()) + sum(v.view(-1).size(0) for k, v in net_glob_server.state_dict().items()))
    print('Client Model parameters: ',sum(v.view(-1).size(0) for k, v in net_glob_client.state_dict().items()))
    main(args)
    endtime = datetime.datetime.now()
    print(endtime-starttime)
    print(args)

