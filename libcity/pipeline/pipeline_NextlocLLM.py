import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
import pandas as pd
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
from accelerate import Accelerator
from torch import nn, optim
import numpy as np
import shutil
from tqdm import tqdm
import time
import datetime
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate.utils import set_seed
from scipy.spatial import KDTree
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import matplotlib
matplotlib.use('Agg')


os.environ["TOKENIZERS_PARALLELISM"] = "false"



def top_k(loc_pred, loc_true, topk,tree,accelerator):
    """
        loc_pred: (batch_size * output_dim)
        loc_true: (batch_size * 1)
        topk:

        tuple: tuple contains:
            hit (int): the hit numbers \n
            dcg (float): dcg
    """
    loc_pred = loc_pred.numpy()
    hit = 0
    result=tree.query(loc_pred,k=topk)[1]
    #batch,topk
    #accelerator.print('result',result,'*',loc_pred)
    for i in range(len(loc_true)):
        #accelerator.print('result in',result[i],loc_true[i])
        if isinstance(result[i], np.int64):
            if(loc_true[i]==result[i]):
                hit+=1
        else:
            if(loc_true[i] in  result[i]):
                hit+=1
    return hit

def vali(config, accelerator, model, vali_loader):
    loc_loss_lst=[]
    dur_loss_lst=[]
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(vali_loader):
            
            #batch= batch_x.to(accelerator.device)
            # 在使用 accelerate 库的情况下正确调用自定义方法
            loc_loss=model.module.calculate_loc_loss(batch,accelerator)
            dur_loss=model.module.calculate_dur_loss(batch,accelerator)
            

            loc_loss_lst.append(loc_loss.item())
            dur_loss_lst.append(dur_loss.item())



    loc_loss_res = np.average(loc_loss_lst)
    dur_loss_res = np.average(dur_loss_lst)

    model.train()

    return loc_loss_res,dur_loss_res

def vali_test_sim(config, accelerator,  model, vali_loader,coordinates):
    loc_loss_lst=[]
    dur_loss_lst=[]
    model.eval()

    hit_rate1=0
    hit_rate5=0
    hit_rate10=0
    hit_rate20=0
    n_sample_valid=len(vali_loader)
    N_v=0
    distance_loss_total=0
    dur_loss=nn.NLLLoss()
    
    #num_loc,dim_vec

    with torch.no_grad():

        coordinates_tensor=torch.FloatTensor(coordinates).to(accelerator.device)

        coordinates_tensor=model.module.normalize_loc(coordinates_tensor,'norm')

        true_loc_emb=model.module.mer2vec(coordinates_tensor)

        for i, batch in enumerate(vali_loader):

            next_loc,_,_=model(batch,accelerator)

            true_loc=batch['target_idx']
            true_loc_=torch.as_tensor(batch['target_idx']).long().to(accelerator.device)
            #next_loc=model.module.mer2vec(next_loc)
            #batch,dim_vec

            similarity=torch.matmul(next_loc,true_loc_emb.t())
            similarity = F.log_softmax(similarity, dim=1)
            distance_loss_total+=dur_loss(similarity,true_loc_)
            #batch,num_loc
            top1_indices = torch.topk(similarity, 1, dim=1).indices
            top5_indices = torch.topk(similarity, 5, dim=1).indices
            top10_indices = torch.topk(similarity, 10, dim=1).indices
            top20_indices = torch.topk(similarity, 20, dim=1).indices
            
            l=len(true_loc)
            N_v+=l

            for idx in range(l):
                if true_loc[idx] in top10_indices[idx]:
                    hit_rate10+=1
                if true_loc[idx] in top5_indices[idx]:
                    hit_rate5+=1
                if true_loc[idx] in top1_indices[idx]:
                    hit_rate1+=1
                if true_loc[idx] in top20_indices[idx]:
                    hit_rate20+=1
            
    model.train()
    accelerator.print('valid_test ',hit_rate1,hit_rate5,hit_rate10,hit_rate20,N_v,
            hit_rate1/N_v,
            hit_rate5/N_v,
            hit_rate10/N_v,
            hit_rate20/N_v,
            (distance_loss_total/N_v)*config['batch_size'],
            accelerator.device)
    return(hit_rate1/N_v,
            hit_rate5/N_v,
            hit_rate10/N_v,
            distance_loss_total/N_v)


def vali_test(config, accelerator,  model, vali_loader,tree):
    loc_loss_lst=[]
    dur_loss_lst=[]
    model.eval()

    hit_rate1=0
    hit_rate5=0
    hit_rate10=0
    hit_rate20=0
    n_sample_valid=len(vali_loader)
    N_v=0
    dist=0
    with torch.no_grad():
        for i, batch in enumerate(vali_loader):
            
            #batch= batch_x.to(accelerator.device)
            # 在使用 accelerate 库的情况下正确调用自定义方法
            
            _,next_loc,_=model(batch,accelerator)
            #accelerator.print('next_loc',next_loc)
            #accelerator.print('tgt',batch['target'])
            true_loc=torch.as_tensor(batch['target']).float().to(accelerator.device)

            dist+=torch.linalg.norm(next_loc-true_loc,dim=1).sum()

            
            next_loc=next_loc.cpu().detach()
            #print('len',len(next_loc[next_loc>0]))

            target=batch['target_idx']
            #print('next_loc',next_loc)
            #print('target',target)




            #accelerator.print('result in vali',next_loc_,'#'*10,next_loc,'@'*10,next_loc_[:,0])
            hit1=top_k(next_loc,target,1,tree,accelerator)
            hit5=top_k(next_loc,target,5,tree,accelerator)
            hit10=top_k(next_loc,target,10,tree,accelerator)
            hit20=top_k(next_loc,target,20,tree,accelerator)
            #print(hit_rate1_batch,hit_rate5_batch,hit_rate10_batch)
            hit_rate1+=hit1
            hit_rate5+=hit5
            hit_rate10+=hit10
            hit_rate20+=hit20

            N_v+=len(batch['target_idx'])
            #print('*'*10)
    model.train()
    accelerator.print('valid_test ',hit_rate1,hit_rate5,hit_rate10,hit_rate20,N_v,
            hit_rate1/N_v,
            hit_rate5/N_v,
            hit_rate10/N_v,
            hit_rate20/N_v,
            dist/N_v,
            accelerator.device)
    return(hit_rate1/N_v,
            hit_rate5/N_v,
            hit_rate10/N_v,
            dist/N_v)

def run_model_NextlocLLM_MER_lora(config_file=None,
                        other_args=None):
    
    #########################    配置加载&设置    #########################
    config = ConfigParser('traj_loc_pred', 
                        'NextlocLLM_MER_lora', 
                        'METR_LA',
                        config_file=config_file,
                        other_args=other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    #logger = get_logger(config)
    #config=config.config
    
    #print(config.config)

    best_val=10000
    #最好的验证loss

    seed = config.get('seed', 0)
    #随机种子设置
    
    set_seed(seed)
    path='./NextlocLLM/libcity/cache/llm/'
    #######################################################################

    

    topk_file = config["topk_file"]
    loc_ = pd.read_csv(topk_file)
    coordinates = loc_[['mercator_x_subzone', 'mercator_y_subzone']].values
    #print('coord',coordinates)
    '''
    [[12947162.07049679  4863015.17602583]
    [12949214.279517    4866146.65995776]
    [12949713.05344352  4864807.14102297]
    ...
    [12950718.66943083  4857258.7764302 ]
    [12947680.11898755  4866494.98623617]
     [12951917.82386099  4846345.21855565]]
    '''
    tree = KDTree(coordinates)
    # validation 相关的 KD树


    #############################  数据集 部分###################################

    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()



    ################################# 模型部分 #############################

    model = get_model(config,data_feature)
    '''
    state_dict = torch.load('./save_file/NextlocLLM.pth')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    '''

    #######################################################################

    



    ##################### 其他配置 ###########################
    accelerator = Accelerator(gradient_accumulation_steps=8)
    #huggingface accelerator
    accelerator.print(model)
    accelerator.print(config.config)
    #early_stopping

    time_now = time.time()

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    for name, param in model.named_parameters():
        if param.requires_grad:
            accelerator.print(f"Parameter: {name}")       
    #需要train的参数【部分参数被冻结了，不用train】

    model_optim = optim.Adam(trained_parameters, 
                            lr=config['learning_rate'],
                            weight_decay=0.01)
    scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=2, verbose=True)
    '''
    train_loader, vali_loader, test_loader, model, model_optim= accelerator.prepare(
            train_data, 
            valid_data, 
            test_data, 
            model, 
            model_optim)
    
    #只有dataspark需要
    scheduler=CyclicLR(
        model_optim,
        base_lr=0.00001,
        max_lr=0.001,
        step_size_up=10,
        step_size_down=10,
        mode='triangular')
    '''
    train_loader, vali_loader, test_loader, model, model_optim= accelerator.prepare(
            train_data, 
            valid_data, 
            test_data, 
            model, 
            model_optim)
    
    train_steps = len(train_loader)

    lst_train_loss=[]
    lst_dis_loss=[]
    lst_dur_loss=[]

    lst_loss_tst=[]
    hit_rate_tst=[]
    lst_loss_val=[]
    lst_loss_tr=[]

    for epoch in range(config['max_epoch']):
        begin=time.time()
        flag=config['max_epoch']-1-epoch
        iter_count=0

        train_loss=[]

        dis_loss=[]
        dur_loss=[]

        model.train()

        epoch_time = time.time()
        
        for i, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                iter_count+=1

                model_optim.zero_grad()

                #batch = batch.to(accelerator.device)

                # 在使用 accelerate 库的情况下正确调用自定义方法
                loss,distance_loss,mse_loss = model.module.calculate_loss(batch,accelerator,flag)
                
                    
                train_loss.append(loss.item())
                dis_loss.append(distance_loss.item())
                dur_loss.append(mse_loss.item())
                accelerator.backward(loss)

                model_optim.step()

               
        scheduler.step(loss)
        
                
        current_lr = model_optim.param_groups[0]['lr']
        accelerator.print("Epoch: {} cost time: {} lr".format(epoch + 1, time.time() - epoch_time,current_lr))
        

        train_loss_ = np.average(train_loss)
        dis_loss_=np.average(dis_loss)
        dur_loss_=np.average(dur_loss)
        lst_train_loss.append(train_loss_)
        lst_dis_loss.append(dis_loss_)
        lst_dur_loss.append(dur_loss_)
        
        #vali_loss_loc, vali_loss_dur = vali(config, accelerator, model, vali_loader)
        if(config['if_sim']):
            accelerator.print('test')
            hit_rate1,hit_rate5,hit_rate10,tst_loss= vali_test_sim(config, accelerator, model, test_loader,coordinates)
            accelerator.print('valid')
            hit_rate1_v,hit_rate5_v,hit_rate10_v,tst_loss_v= vali_test_sim(config, accelerator, model, vali_loader,coordinates)
            accelerator.print('train')
            hit_rate1_t,hit_rate5_tr,hit_rate10_tr,tst_loss_tr = vali_test_sim(config, accelerator, model, train_loader,coordinates)
        else:
            accelerator.print('test')
            hit_rate1,hit_rate5,hit_rate10,tst_loss= vali_test(config, accelerator, model, test_loader,tree)
            accelerator.print('valid')
            hit_rate1_t,hit_rate5_v,hit_rate10_v,tst_loss_v = vali_test(config, accelerator, model, vali_loader,tree)
            accelerator.print('train')
            hit_rate1_t,hit_rate5_tr,hit_rate10_tr,tst_loss_tr = vali_test(config, accelerator, model, train_loader,tree)
        
        if accelerator.is_main_process:
            if(hit_rate10_v<best_val):
                best_val=hit_rate10_v
                unwarp_model=accelerator.unwrap_model(model)
                torch.save(model.state_dict(),'./NextlocLLM/save_file/NextlocLLM.pth')
            


        #hit_rate1_c,hit_rate5_c,hit_rate10_c,tst_loss_c = vali_test_coord(config, accelerator, model, test_loader,coordinates)
        tst_loss_=tst_loss.cpu()
        tst_loss_v_=tst_loss_v.cpu()
        tst_loss_tr_=tst_loss_tr.cpu()
        
        lst_loss_tst.append(tst_loss_)
        hit_rate_tst.append(hit_rate10)
        lst_loss_val.append(tst_loss_v_)
        lst_loss_tr.append(tst_loss_tr_)
        
        accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} test_hit_rate_1: {2:.7f} test_hit_rate_5:{3:.7f} test_hit_rate_10:{4:.7f} distance loss:{5:.7f} duraction loss:{6:.7f} hit rate train:{7:.7f}".format(
                    epoch + 1, train_loss_, hit_rate1,hit_rate5,hit_rate10,dis_loss_,dur_loss_,hit_rate10_tr))
        accelerator.print(time.time()-begin)
        

    
    if(config['if_sim']):
                hit_rate1,hit_rate5,hit_rate10,tst_loss = vali_test_sim(config, accelerator, model, test_loader,coordinates)
    else:
                hit_rate1,hit_rate5,hit_rate10,tst_loss = vali_test(config, accelerator, model, test_loader,tree)
    print("Finally test_hit_rate_1: {0:.7f} test_hit_rate_5:{1:.7f} test_hit_rate_10:{2:.7f} ".format(
                    hit_rate1,hit_rate5,hit_rate10,model.device))
    
    

def test_model_NextlocLLM_MER_lora(config_file=None,
                        other_args=None):
    
    #########################    配置加载&设置    #########################
    config = ConfigParser('traj_loc_pred', 
                        'NextlocLLM_MER_lora', 
                        'METR_LA',
                        config_file=config_file,
                        other_args=other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    #config=config.config
    #print(config)

    # seed

    seed = config.get('seed', 0)
    
    set_seed(seed)
    path='~/Bigscity-LibCity/libcity/cache/llm/'
    #######################################################################

    

    topk_file = config["topk_file"]
    loc_ = pd.read_csv(topk_file)
    coordinates = loc_[['mercator_x_subzone', 'mercator_y_subzone']].values
    #print(coordinates.shape)
    tree = KDTree(coordinates)
    # validation 相关的 KD树


    #############################  数据集 部分###################################

    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()



    ################################# 模型部分 #############################

    model = get_model(config,data_feature)
    
    '''
    #######################################################################
    state_dict = torch.load('./save_file/NextlocLLM.pth')
    #print(state_dict.keys())
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    '''
    


    ##################### 其他配置 ###########################
    accelerator = Accelerator()#gradient_accumulation_steps=2)
    #huggingface accelerator
    accelerator.print(model)

    accelerator.print(config.config)

    #early_stopping

    time_now = time.time()

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
    '''
    for name,p in model.named_parameters():
        if p.requires_grad is True:
            print(name)
    '''
    #需要train的参数【部分参数被冻结了，不用train】

    #print('learning_rate',config['learning_rate'])
    model_optim = optim.Adam(trained_parameters, 
                            lr=config['learning_rate'])
    train_loader, vali_loader, test_loader, model, model_optim= accelerator.prepare(
            train_data, 
            valid_data, 
            test_data, 
            model, 
            model_optim)
    
    accelerator.load_state(config["save_dir"])
    train_steps = len(train_loader)

    loc_loss_lst=[]
    dur_loss_lst=[]
    model.eval()


    with torch.no_grad():
            if(config['if_sim']):
                hit_rate1,hit_rate5,hit_rate10,tst_loss = vali_test_sim(config, accelerator, model, test_loader,coordinates)
            else:
                hit_rate1,hit_rate5,hit_rate10,tst_loss = vali_test(config, accelerator, model, test_loader,tree)
            print(
                "Finally test_hit_rate_1: {0:.7f} test_hit_rate_5:{1:.7f} test_hit_rate_10:{2:.7f} ".format(
                    hit_rate1,hit_rate5,hit_rate10))
        
