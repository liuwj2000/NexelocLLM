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
    file_dir='../data/gridize_notebooks/CHENGDU/llmmob/history_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data

def load_content_data(name,type):
    file_dir='../data/gridize_notebooks/CHENGDU/llmmob/context_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data

def load_content_data_true(name,type):
    file_dir='../data/gridize_notebooks/CHENGDU/llmmob/context_true_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data


class ChengduNextlocLLMDatasetMer(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.train_data_dir=''
        self.logger = getLogger()

    def get_data(self):
        

        #####################################################            load history data and related info       #####################################################

        history_data_train=load_history_data('data','train')
        history_data_subzone_train=load_history_data('data_subzone','train')
        history_subzone_id_train=load_history_data('subzone_id','train')
        history_dur_train=load_history_data('dur','train')
        history_hour_train=load_history_data('hour','train')
        history_day_train=load_history_data('day','train')
        history_poi_train=load_history_data('poi','train')        

        history_data_valid=load_history_data('data','vali')
        history_data_subzone_valid=load_history_data('data_subzone','vali')
        history_subzone_id_valid=load_history_data('subzone_id','vali')
        history_dur_valid=load_history_data('dur','vali')
        history_hour_valid=load_history_data('hour','vali')
        history_day_valid=load_history_data('day','vali')
        history_poi_valid=load_history_data('poi','vali')  

        history_data_test=load_history_data('data','test')
        history_data_subzone_test=load_history_data('data_subzone','test')
        history_subzone_id_test=load_history_data('subzone_id','test')
        history_dur_test=load_history_data('dur','test')
        history_hour_test=load_history_data('hour','test')
        history_day_test=load_history_data('day','test')
        history_poi_test=load_history_data('poi','test')  

        ###############################################################################################################################################################
        
        


        #####################################################            load content data and related info   for training     #####################################################
        context_data_train=load_content_data('data','train')
        context_dur_train=load_content_data('dur','train')
        context_data_subzone_train=load_content_data('data_subzone','train')
        context_subzone_id_train=load_content_data('subzone_id','train')
        context_hour_train=load_content_data('hour','train')
        context_day_train=load_content_data('day','train')
        context_poi_train=load_content_data('poi','train')
        
        ###############################################################################################################################################################
        




        #####################################################            load content data and related info   for training     #####################################################
        context_data_valid=load_content_data('data','vali')
        context_dur_valid=load_content_data('dur','vali')
        context_data_subzone_valid=load_content_data('data_subzone','vali')
        context_subzone_id_valid=load_content_data('subzone_id','vali')
        context_hour_valid=load_content_data('hour','vali')
        context_day_valid=load_content_data('day','vali')
        context_poi_valid=load_content_data('poi','vali')
        ###############################################################################################################################################################
        



        #####################################################            load content data and related info   for training     #####################################################
        context_data_test=load_content_data('data','test')
        context_dur_test=load_content_data('dur','test')
        context_data_subzone_test=load_content_data('data_subzone','test')
        context_subzone_id_test=load_content_data('subzone_id','test')
        context_hour_test=load_content_data('hour','test')
        context_day_test=load_content_data('day','test')
        context_poi_test=load_content_data('poi','test')
        ###############################################################################################################################################################





        ####################################################            load ground truth data  ############################################################

        context_data_train_true=load_content_data_true('data','train')
        context_data_subzone_train_true=load_content_data_true('data_subzone','train')
        context_dur_train_true=load_content_data_true('dur','train')
        context_data_idx_train_true=load_content_data_true('data_idx','train')

        context_data_valid_true=load_content_data_true('data','vali')
        context_data_subzone_valid_true=load_content_data_true('data_subzone','vali')
        context_dur_valid_true=load_content_data_true('dur','vali')
        context_data_idx_valid_true=load_content_data_true('data_idx','vali')

        context_data_test_true=load_content_data_true('data','test')
        context_data_subzone_test_true=load_content_data_true('data_subzone','test')
        context_dur_test_true=load_content_data_true('dur','test')
        context_data_idx_test_true=load_content_data_true('data_idx','test')
        #####################################################################################################################################################
        
        
        
        
        
        ###########################################   train_data ebegin ###########################################
        train_data_llm=[]
        for user in history_data_train:
            len_user=len(context_data_train[user])
            for i in range(1,len_user):
                tmp=[]
                tmp.append(history_data_train[user][i])
                tmp.append(history_data_subzone_train[user][i])
                tmp.append(history_subzone_id_train[user][i])
                tmp.append(history_day_train[user][i])
                tmp.append(history_hour_train[user][i])
                tmp.append(history_dur_train[user][i])
                tmp.append(history_poi_train[user][i])
                
                tmp.append(context_data_train[user][i])
                tmp.append(context_data_subzone_train[user][i])
                tmp.append(context_subzone_id_train[user][i])
                tmp.append(context_day_train[user][i])
                tmp.append(context_hour_train[user][i])
                tmp.append(context_dur_train[user][i])
                tmp.append(context_poi_train[user][i])
                
                tmp.append(context_data_train_true[user][i])
                tmp.append(context_data_subzone_train_true[user][i])
                #try:
                #  print('contect_data',user,'index',i,len(context_data_train_true[user][i]))
                #except:
                #  pass
                tmp.append(context_dur_train_true[user][i])
                tmp.append(context_data_idx_train_true[user][i])
                tmp.append(user)
                train_data_llm.append(tmp)
                #print(tmp)

        self.logger.info('Num of training data:{}'.format(len(train_data_llm)))
        ###########################################   train_data end ###########################################




        ###########################################   valid_data begin ###########################################
        valid_data_llm=[]
        for user in history_data_valid:
            len_user=len(context_data_valid[user])
            for i in range(1,len_user):
                tmp=[]
                tmp.append(history_data_valid[user][i])
                tmp.append(history_data_subzone_valid[user][i])
                tmp.append(history_subzone_id_valid[user][i])
                tmp.append(history_day_valid[user][i])
                tmp.append(history_hour_valid[user][i])
                tmp.append(history_dur_valid[user][i])
                tmp.append(history_poi_valid[user][i])
                
                tmp.append(context_data_valid[user][i])
                tmp.append(context_data_subzone_valid[user][i])
                tmp.append(context_subzone_id_valid[user][i])
                tmp.append(context_day_valid[user][i])
                tmp.append(context_hour_valid[user][i])
                tmp.append(context_dur_valid[user][i])
                tmp.append(context_poi_valid[user][i])
                
                tmp.append(context_data_valid_true[user][i])
                tmp.append(context_data_subzone_valid_true[user][i])
                tmp.append(context_dur_valid_true[user][i])
                tmp.append(context_data_idx_valid_true[user][i])
                tmp.append(user)
                valid_data_llm.append(tmp)
                #print(tmp)

        self.logger.info('Num of validation data:{}'.format(len(valid_data_llm)))
        ###########################################   valid_data end ###########################################




        ###########################################   test_data begin ###########################################
        test_data_llm=[]
        for user in history_data_test:
            len_user=len(context_data_test[user])
            for i in range(len_user):
                tmp=[]
                tmp.append(history_data_test[user][i])
                tmp.append(history_data_subzone_test[user][i])
                tmp.append(history_subzone_id_test[user][i])
                tmp.append(history_day_test[user][i])
                tmp.append(history_hour_test[user][i])
                tmp.append(history_dur_test[user][i])
                tmp.append(history_poi_test[user][i])
                
                tmp.append(context_data_test[user][i])
                tmp.append(context_data_subzone_test[user][i])
                tmp.append(context_subzone_id_test[user][i])
                tmp.append(context_day_test[user][i])
                tmp.append(context_hour_test[user][i])
                tmp.append(context_dur_test[user][i])
                tmp.append(context_poi_test[user][i])
                
                tmp.append(context_data_test_true[user][i])
                tmp.append(context_data_subzone_test_true[user][i])
                tmp.append(context_dur_test_true[user][i])
                tmp.append(context_data_idx_test_true[user][i])
                tmp.append(user)
                test_data_llm.append(tmp)
                #print(tmp)

        self.logger.info('Num of testing data:{}'.format(len(test_data_llm)))
        ###########################################   test_data end ###########################################
        

        self.loc_x_mean=11585630.89506371
        self.loc_y_mean=3591489.968417769

        self.loc_x_std=2272.7599708069397
        self.loc_y_std=2288.231009240483
        self.duration_max=62.41666666666666
        self.duration_min=0.333

        feature_dict={'history_loc':'array of int',
                      'history_subzone_loc':'array of int',
                      'history_subzone_id':'int',
                      'history_day':'int',
                      'history_hour':'int',
                      'history_dur':'int',
                      'history_poi':'array of int',
                      'current_loc':'array of int',
                      'current_subzone_loc':'array of int',
                      'current_subzone_id':'int',
                      'current_day':'int',
                      'current_hour':'int',
                      'current_dur':'int',
                      'current_poi':'array of int',
                      'target'     :'array of int',
                      'target_subzone'     :'array of int',
                      'target_dur' :'int',
                      'target_idx':'int',
                      'uid'        :'int'}

        pad_item={}
        '''
        pad_item={'history_loc':[self.loc_x_mean,self.loc_y_mean],
                  'history_day':7,
                  'history_hour':24,
                  'history_dur':0,
                  'history_poi':[0,0,0,0,0],
                  'current_loc':[self.loc_x_mean,self.loc_y_mean],
                  'current_day':7,
                  'current_hour':24,
                  'current_dur':0,
                  'current_poi':[0,0,0,0,0]
                  }
        '''
        # num of stations 

  
        

        
        #train_data, eval_data, test_data = self.divide_data()
        train_dataloader,valid_data_loader,test_dataloader=generate_dataloader_pad(train_data_llm, 
                                       valid_data_llm,     
                                       test_data_llm,
                                       feature_dict,
                                       self.config['batch_size'],
                                       self.config['num_workers'], 
                                       pad_item
                                       )
        '''
        for batch in train_dataloader:
            history_data=np.array(batch['history_loc'])
            history_dur=np.array(batch['history_dur'])
            history_hour=np.array(batch['history_hour'])
            history_day=np.array(batch['history_day'])
            history_poi=np.array(batch['history_poi'])

            current_data=np.array(batch['current_loc'])
            current_dur=np.array(batch['current_dur'])
            current_hour=np.array(batch['current_hour'])
            current_day=np.array(batch['current_day'])
            current_poi=np.array(batch['current_poi'])
            
            target=np.array(batch['target'])
            target_idx=np.array(batch['target_idx'])

            print(history_data.shape,history_dur.shape,history_hour.shape,history_day.shape,history_poi.shape)
            print(current_data.shape,current_dur.shape,current_hour.shape,current_day.shape,current_poi.shape)
            print('*'*100)
            
            for i in ['history_data','history_dur','history_hour','history_day','history_poi','current_data','current_dur','current_hour','current_day','current_poi','target','target_idx']:
                np.save('../data/geolife/test/'+i+'_mer.npy',eval(i))
            break
        '''
            
        return train_dataloader,valid_data_loader,test_dataloader

    def get_data_feature(self):
        res = {'loc_size':  332,
               'day_size':  7,
               'hour_size': 24,
               'dur_size':  1402,
               'uid_size':  3392,
               'loc_pad':  907,
               'hour_pad':  24,
               'day_pad':7,
               'dur_pad':0,
               'loc_x_mean':self.loc_x_mean,
               'loc_y_mean':self.loc_y_mean,
               'loc_x_std':self.loc_x_std,
               'loc_y_std':self.loc_y_std,
               'duration_max':self.duration_max,
               'duration_min':self.duration_min}
        #res['distance_upper'] = self.config['distance_upper']
        return res

    
