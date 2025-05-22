import os
import json
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
import importlib
from logging import getLogger
import pickle

from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_timeoff
from libcity.data.utils import generate_dataloader_pad

parameter_list = ['dataset', 'min_session_len', 'min_sessions', "max_session_len",
                  'cut_method', 'window_size', 'min_checkins']


def load_history_data(name,type):
    file_dir='../data/gridize_notebooks/XIAN/llmmob/history_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data

def load_content_data(name,type):
    file_dir='../data/gridize_notebooks/XIAN/llmmob/context_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data

def load_content_data_true(name,type):
    file_dir='../data/gridize_notebooks/XIAN/llmmob/context_true_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data


class XianBaselineDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.train_data_dir=''
        self.logger = getLogger()

    def get_data(self):
        
        #####################################################            load history data and related info       #####################################################

        history_data_train=load_history_data('subzone_id','train')
        history_hour_train=load_history_data('hour','train')      

        history_data_valid=load_history_data('subzone_id','vali')
        history_hour_valid=load_history_data('hour','vali')

        history_data_test=load_history_data('subzone_id','test')
        history_hour_test=load_history_data('hour','test')

        ###############################################################################################################################################################
        
        context_data_train=load_content_data('subzone_id','train')
        context_hour_train=load_content_data('hour','train')      

        context_data_valid=load_content_data('subzone_id','vali')
        context_hour_valid=load_content_data('hour','vali')

        context_data_test=load_content_data('subzone_id','test')
        context_hour_test=load_content_data('hour','test')
        ###############################################################################################################################################################





        ####################################################            load ground truth data  ############################################################

        context_data_train_true=load_content_data_true('data_idx','train')
        context_hour_train_true=load_content_data_true('hour','train')

        context_data_valid_true=load_content_data_true('data_idx','vali')
        context_hour_valid_true=load_content_data_true('hour','vali')

        context_data_test_true=load_content_data_true('data_idx','test')
        context_hour_test_true=load_content_data_true('hour','test')
        #####################################################################################################################################################
        
        
        
        
        
        train_data=[]
        for user in history_data_train:
            len_user=len(context_data_train[user])
            for i in range(len_user):
                tmp=[]
                tmp.append(history_data_train[user][i])
                tmp.append(history_hour_train[user][i])
                
                tmp.append(context_data_train[user][i])
                tmp.append(context_hour_train[user][i])
                
                tmp.append(context_data_train_true[user][i])
                tmp.append(context_hour_train_true[user][i])
                tmp.append(user)
                train_data.append(tmp)

        self.logger.info('Num of training data:{}'.format(len(train_data)))


        valid_data=[]
        for user in history_data_valid:
            len_user=len(context_data_valid[user])
            for i in range(len_user):
                tmp=[]
                tmp.append(history_data_valid[user][i])
                tmp.append(history_hour_valid[user][i])
                
                tmp.append(context_data_valid[user][i])
                tmp.append(context_hour_valid[user][i])
                
                tmp.append(context_data_valid_true[user][i])
                tmp.append(context_hour_valid_true[user][i])
                tmp.append(user)
                valid_data.append(tmp)

        self.logger.info('Num of training data:{}'.format(len(valid_data)))

        test_data=[]
        for user in history_data_test:
            len_user=len(context_data_test[user])
            for i in range(len_user):
                tmp=[]
                tmp.append(history_data_test[user][i])
                tmp.append(history_hour_test[user][i])
                
                tmp.append(context_data_test[user][i])
                tmp.append(context_hour_test[user][i])
                
                tmp.append(context_data_test_true[user][i])
                tmp.append(context_hour_test_true[user][i])
                tmp.append(user)
                test_data.append(tmp)

        self.logger.info('Num of training data:{}'.format(len(test_data)))

        feature_dict={'history_loc':'int',
                      'history_tim':'int',
                      'current_loc':'int',
                      'current_tim':'int',
                      'target'     :'int',
                      'target_tim' :'int',
                      'uid'        :'int'}


        pad_item={'current_loc':943,
                  'history_loc':943,
                  'current_tim':48,
                  'history_tim':48}


  
        

        
        return generate_dataloader_pad(train_data, 
                                       valid_data,     
                                       test_data,
                                       feature_dict,
                                       self.config['batch_size'],
                                       self.config['num_workers'], 
                                       pad_item
                                       )
    def get_data_feature(self):
        res = {'loc_size':  944,
               'tim_size': 49,
               'uid_size':  1000000,
               'loc_pad':  943,
               'tim_pad':  48}
        return res

    
