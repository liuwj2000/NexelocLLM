import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from logging import getLogger
from torch.utils.checkpoint import checkpoint
from libcity.model.abstract_model import AbstractModel
import pandas as pd
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model




from transformers import AutoModel,AutoTokenizer
#torch.set_printoptions(threshold=5000)

os.environ["HF_TOKEN"] = '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'

def normalize_rows(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)  # 计算每行的L2范数
    return x / (norm + 1e-6)  # 防止除零错误

class POIMLP(nn.Module):
    def __init__(self,llm_dim,max_len_poi):
        super(POIMLP, self).__init__()
        #self.fc1 = nn.Linear(17*768, 1024,bias=False)  # 第一层
        #self.fc2 = nn.Linear(1024, 768,bias=False)     # 第二层
        self.fc1 = nn.Linear(max_len_poi*llm_dim, 1024,bias=False)
        self.fc2 = nn.Linear(1024, llm_dim,bias=False)

    def forward(self, x):
        #x = x.view(-1, 17*768)  # 展平输入以匹配第一层的维度
        x=x.view(-1,max_len_poi*llm_dim)
        x = F.tanh(self.fc1(x)) # 第一层 + ReLU 激活
        x = self.fc2(x)         # 第二层
        return x

class Normalize_loc(nn.Module):
    def __init__(self,loc_x_mean,loc_y_mean,loc_x_std,loc_y_std):
        super(Normalize_loc,self).__init__()
        self.loc_x_mean=loc_x_mean
        self.loc_y_mean=loc_y_mean
        self.loc_x_std=loc_x_std
        self.loc_y_std=loc_y_std
        self.loc_mean=torch.FloatTensor([self.loc_x_mean,self.loc_y_mean])
        self.loc_std=torch.FloatTensor([self.loc_x_std,self.loc_y_std])
        #print(self.loc_mean,self.loc_std)
    def forward(self,x,mode):
        if mode=='norm':
            x=self.normalize(x)
        elif mode=='denorm':
            x=self.denormalize(x)
        return x

    def normalize(self,x):
        self.loc_mean=self.loc_mean.to(x.device)
        self.loc_std=self.loc_std.to(x.device)
        x=(x-self.loc_mean)/self.loc_std
        return x
    
    def denormalize(self,x):
        self.loc_mean=self.loc_mean.to(x.device)
        self.loc_std=self.loc_std.to(x.device)
        x=x*self.loc_std+self.loc_mean
        return x 

class Normalize_dur(nn.Module):
    def __init__(self,duration_max,duration_min):
        super(Normalize_dur,self).__init__()
        self.duration_max=torch.FloatTensor([duration_max])
        self.duration_min=torch.FloatTensor([duration_min])
        #print(self.duration_max,self.duration_min)
    def forward(self,x,mode):
        if mode=='norm':
            x=self.normalize(x)
        elif mode=='denorm':
            x=self.denormalize(x)
        return x

    def normalize(self,x):
        self.duration_min=self.duration_min.to(x.device)
        self.duration_max=self.duration_max.to(x.device)
        x=(x-self.duration_min)/(self.duration_max-self.duration_min)
        return x
    
    def denormalize(self,x):
        self.duration_min=self.duration_min.to(x.device)
        self.duration_max=self.duration_max.to(x.device)
        x=x*(self.duration_max-self.duration_min)+self.duration_min
        return x

class NextlocLLM_MER_lora(AbstractModel):
    """rnn model with long-term history attention"""



    def __init__(self, config, data_feature):
        super(NextlocLLM_MER_lora, self).__init__(config, data_feature) 

        

        self.dropout = nn.Dropout(p=config['dropout_p'])
        #print('in_file ')
        #print(config)
        self.if_dur_loss=config['if_dur_loss']
        topk_file = config["topk_file"]
        loc_ = pd.read_csv(topk_file)
        self.coordinates = loc_[['mercator_x_subzone', 'mercator_y_subzone']].values
        self.if_dur_emb=config['if_dur_emb']
        self.if_sim=config['if_sim']
        self.if_lora=config['if_lora']
        self.if_llm_poi=config['if_llm_poi']
        self.if_poi=config['if_poi']
        self.if_prompt=config['if_prompt']
        self.lambda_loss=config['lambda_loss']

        self.dim_size=config['dim_feature']

        #data location embedding
        self.mer_size = 2
        self.mer_dim=config['mer_dim']
        self.mer2vec=nn.Linear(self.mer_size,self.mer_dim,bias=False)
        #(batch,seq_len,2)——>(batch,seq_len,mervec_size)

        
        #day embedding size
        self.day_size = data_feature['day_size']
        self.day_dim=config['day_dim']
        self.day_embedding=nn.Embedding(self.day_size,
                                        self.day_dim)
        #(batch,seq_len)——>(batch,seq_len,7)


        
        #hour embedding 
        self.hour_size=data_feature['hour_size']
        self.hour_dim=config['hour_dim']
        self.hour_embedding=nn.Embedding(self.hour_size,
                                        self.hour_dim)
        #(batch,seq_len)——>(batch,seq_len,hour_emb_size)
        
        self.dur_dim=config['dur_dim']
        self.dur_linear=nn.Linear(1,self.dur_dim,bias=False)
        
        self.poi_dim=config['poi_dim']
        self.poi_linear=nn.Linear(5,self.poi_dim,bias=False)


        

        self.normalize_loc=Normalize_loc(data_feature['loc_x_mean'],
                                        data_feature['loc_y_mean'],
                                        data_feature['loc_x_std'],
                                        data_feature['loc_y_std'])

        self.normalize_dur=Normalize_dur(data_feature['duration_max'],
                                        data_feature['duration_min'])

        self.logger = getLogger()


        ###llm part
        if(config['llm_model']=='Llama-3-8B'):
            
            self.tokenizer=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
            self.model=AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B',
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True) 
            self.model.layers=self.model.layers[:config['num_layer']]  
            self.max_len_poi=30                                   
        elif(config['llm_model']=='Llama-2-7b'):
            self.tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
            self.model=AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True)
            self.model.layers=self.model.layers[:config['num_layer']]
            self.max_len_poi=30
        elif(config['llm_model']=='gpt2'):
            self.tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
            self.model=AutoModel.from_pretrained('openai-community/gpt2',
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True)

            self.model.h=self.model.h[:config['num_layer']]
            self.max_len_poi=17
            #print(self.model.get_memory_footprint())
        
        self.llm_dim=self.model.get_input_embeddings().weight.shape[1]
        if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token
                
        #################################     POI embedding    ###################################
        #print()
        if(config['if_llm_poi']==True):
            self.poi_mlp=POIMLP(self.llm_dim,self.max_len_poi)
            self.Entertainment='Entertainment: This category combines scenic spots with sports and recreation services for leisure activities.'
            self.Commercial='Commercial: It includes businesses, financial services, automotive, shopping, and dining services.'
            self.Education='Education: This category covers institutions which involved in science, education, and cultural services.'
            self.Public='Public Service: including government, daily services, healthcare, transport, and public infrastructure.'
            self.Residential='Residential: This category comprises accommodation services and mixed-use commercial and residential areas.'

        
            with torch.no_grad():
                if(config['llm_model']!='gpt2'):
                    en_emb=self.tokenizer(self.Entertainment,
                            return_tensors="pt",
                            padding='max_length', 
                            max_length=self.max_len_poi,
                            truncation=True).input_ids
                    #en_emb_llm=self.model.wte(en_emb)
                    en_emb_llm=self.model.embed_tokens(en_emb)
                    
                    c_emb=self.tokenizer(self.Commercial,
                            return_tensors="pt",
                            padding='max_length',  
                            max_length=self.max_len_poi, 
                            truncation=True).input_ids
                    #c_emb_llm=self.model.wte(c_emb)
                    c_emb_llm=self.model.embed_tokens(c_emb)
                    
                    ed_emb=self.tokenizer(self.Education,
                            return_tensors="pt", 
                            padding='max_length', 
                            max_length=self.max_len_poi, 
                            truncation=True).input_ids
                    #ed_emb_llm=self.model.wte(ed_emb)
                    ed_emb_llm=self.model.embed_tokens(ed_emb)

                    p_emb=self.tokenizer(self.Public,
                            return_tensors="pt", 
                            padding='max_length', 
                            max_length=self.max_len_poi, 
                            truncation=True).input_ids
                    #p_emb_llm=self.model.wte(p_emb)
                    p_emb_llm=self.model.embed_tokens(p_emb)
                    
                    r_emb=self.tokenizer(self.Residential,
                            return_tensors="pt", 
                            padding='max_length', 
                            max_length=self.max_len_poi, 
                            truncation=True).input_ids
                    #r_emb_llm=self.model.wte(r_emb)
                    r_emb_llm=self.model.embed_tokens(r_emb)

                    self.vectors = torch.stack([en_emb_llm, 
                            c_emb_llm, 
                            ed_emb_llm, 
                            p_emb_llm, 
                            r_emb_llm], 
                            dim=0)
                            
                    #torch.Size([5, 1, 17, 768])

                    self.vectors=self.vectors.squeeze()
                    #torch.Size([5, 17, 768])

                    self.vectors = self.vectors.unsqueeze(0).unsqueeze(0)
                    #torch.Size([1, 1, 5, 17, 768])
                else:
                    en_emb=self.tokenizer(self.Entertainment,
                            return_tensors="pt").input_ids
                    #en_emb_llm=self.model.wte(en_emb)
                    en_emb_llm=self.model.wte(en_emb)
                    
                    c_emb=self.tokenizer(self.Commercial,
                            return_tensors="pt").input_ids
                    #c_emb_llm=self.model.wte(c_emb)
                    c_emb_llm=self.model.wte(c_emb)
                    
                    ed_emb=self.tokenizer(self.Education,
                            return_tensors="pt").input_ids
                    #ed_emb_llm=self.model.wte(ed_emb)
                    ed_emb_llm=self.model.wte(ed_emb)

                    p_emb=self.tokenizer(self.Public,
                            return_tensors="pt").input_ids
                    #p_emb_llm=self.model.wte(p_emb)
                    p_emb_llm=self.model.wte(p_emb)
                    
                    r_emb=self.tokenizer(self.Residential,
                            return_tensors="pt").input_ids
                    #r_emb_llm=self.model.wte(r_emb)
                    r_emb_llm=self.model.wte(r_emb)

                    self.vectors = torch.stack([en_emb_llm, 
                            c_emb_llm, 
                            ed_emb_llm, 
                            p_emb_llm, 
                            r_emb_llm], 
                            dim=0)
                            
                    #torch.Size([5, 1, 17, 768])

                    self.vectors=self.vectors.squeeze()
                    #torch.Size([5, 17, 768])

                    self.vectors = self.vectors.unsqueeze(0).unsqueeze(0)
                    #torch.Size([1, 1, 5, 17, 768])
            self.history_llmize_linear=nn.Linear(self.mer_dim+self.day_dim+self.dur_dim+self.hour_dim,
                                                self.llm_dim,
                                                bias=False)
            self.current_llmize_linear=nn.Linear(self.mer_dim+self.day_dim+self.dur_dim+self.hour_dim,
                                                self.llm_dim,
                                                bias=False)
            
            self.his_additional_poi=nn.Linear(self.max_len_poi*self.llm_dim,self.llm_dim)
            self.cur_additional_poi=nn.Linear(self.max_len_poi*self.llm_dim,self.llm_dim)

            self.batch_norm = nn.BatchNorm1d(num_features=self.mer_dim+self.day_dim+self.dur_dim+self.hour_dim)
        else:
            
            self.history_llmize_linear=nn.Linear(self.mer_dim+self.day_dim+self.dur_dim+self.hour_dim+self.poi_dim,
                                                self.llm_dim,
                                                bias=False)
            self.current_llmize_linear=nn.Linear(self.mer_dim+self.day_dim+self.dur_dim+self.hour_dim+self.poi_dim,
                                                self.llm_dim,
                                                bias=False)
            self.batch_norm = nn.BatchNorm1d(num_features=self.mer_dim+self.day_dim+self.dur_dim+self.hour_dim+self.poi_dim)
            
        ##########################################################################################


        

        


        
        #print(self.model)
        
        
        #use lora
        if self.if_lora==False:
            for param in self.model.parameters():
                param.requires_grad=False
            
            for name, param in self.model.named_parameters():
                if(config['llm_model']=='gpt2'):
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                else:
                    if "input_layernorm" in name:
                        param.requires_grad = True
            
        else:
            self.peft_config = LoraConfig(
                         r=8, 
                         lora_alpha=32, 
                         lora_dropout=0.1)
                         
        
            self.model = get_peft_model(self.model, self.peft_config)
        

        #self.output_dur=nn.Linear(self.llm_dim,1,bias=False)
        self.output_dur=nn.Linear(self.llm_dim,1,bias=False)

        if(config['if_sim']):
            self.output_loc=nn.Linear(self.llm_dim,self.mer_dim,bias=False)
        else:
            self.output_loc=nn.Linear(self.llm_dim,2,bias=False)

        self.his_seq_len=40
        self.cur_seq_len=5

        #self.position_embedding = PositionalEmbedding(d_model=self.llm_dim)

        #self.init_norm()
        #only dataspark needs


    def init_norm(self):
        nn.init.xavier_uniform_(self.mer2vec.weight)   
        nn.init.xavier_uniform_(self.day_embedding.weight)   
        nn.init.xavier_uniform_(self.hour_embedding.weight)   
        nn.init.xavier_uniform_(self.dur_linear.weight)   
        nn.init.xavier_uniform_(self.poi_linear.weight)   
        nn.init.xavier_uniform_(self.history_llmize_linear.weight)   
        nn.init.xavier_uniform_(self.current_llmize_linear.weight)  

        # 如果这些层有偏置项（bias），并且你也想初始化它们，可以使用以下代码：
        '''
        if self.mer2vec.bias is not None:
            nn.init.zeros_(self.mer2vec.bias)
        if self.dur_linear.bias is not None:
            nn.init.zeros_(self.dur_linear.bias)
        if self.poi_linear.bias is not None:
            nn.init.zeros_(self.poi_linear.bias)
        if self.history_llmize_linear.bias is not None:
            nn.init.zeros_(self.history_llmize_linear.bias)
        if self.current_llmize_linear.bias is not None:
            nn.init.zeros_(self.current_llmize_linear.bias)
        '''


    def forward(self,batch,accelerator=None):
        
        next_vec=self.predict(batch,accelerator)
        #batch,prompt_seq_len+his_seq_len+cur_seq_len,llm_dim
        #next_loc_emb=next_vec[:,-1,:self.mervec_size]
        #accelerator.print('next_vec',next_vec.shape,next_vec)
        next_loc_emb=next_vec[:,-1,:]
        #batch,dim_for_loc
        next_dur_emb=next_vec[:,-1,:]
        #batch,dim_for_dur
        #next_dur_emb=next_vec[:,-1,-self.dim_for_dur:]
        #accelerator.print('next_loc_emb',next_loc_emb.shape,next_loc_emb)
        next_loc=self.predict_loc(next_loc_emb,accelerator)
        #accelerator.print('next_loc',next_loc.shape,next_loc)
        #batch,loc_size
        #next_loc_denorm=next_loc
        
        #accelerator.print('next_loc_denorm',next_loc_denorm.shape,next_loc_denorm)

        next_dur=self.predict_dur(next_dur_emb,accelerator)
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        #batch,1

        next_dur=self.normalize_dur(next_dur,'denorm')
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        if(self.if_sim==False):
            next_loc_denorm=self.normalize_loc(next_loc,'denorm')
            #print('next_loc_denorm',next_loc_denorm[0])
            return next_loc,next_loc_denorm,next_dur
        else:
            return next_loc,None,next_dur


    def predict(self, batch,accelerator=None):

        ################################load batch data ########################
        
        history_data=batch['history_loc']
        #history_data=batch['history_subzone_loc']
        history_dur=batch['history_dur']
        history_hour=batch['history_hour']
        history_day=batch['history_day']
        history_poi=batch['history_poi']

        current_data=batch['current_loc']
        #current_data=batch['current_subzone_loc']
        current_dur=batch['current_dur']
        current_hour=batch['current_hour']
        current_day=batch['current_day']
        current_poi=batch['current_poi']

        uer_id=batch['uid']

        batch_num=len(batch['current_poi'])

        
        history_data_t=torch.as_tensor(history_data).float()[:,-self.his_seq_len:].to(accelerator.device)
        history_dur_t=torch.as_tensor(history_dur).float()[:,-self.his_seq_len:].to(accelerator.device)
        history_hour_t=torch.as_tensor(history_hour).long()[:,-self.his_seq_len:].to(accelerator.device)
        history_day_t=torch.as_tensor(history_day).long()[:,-self.his_seq_len:].to(accelerator.device)
        history_poi_t=torch.as_tensor(history_poi).float()[:,-self.his_seq_len:].to(accelerator.device)
        #print('before norm',history_dur_t.shape)

        current_data_t=torch.as_tensor(current_data).float().to(accelerator.device)
        current_dur_t=torch.as_tensor(current_dur).float().to(accelerator.device)
        current_hour_t=torch.as_tensor(current_hour).long().to(accelerator.device)
        current_day_t=torch.as_tensor(current_day).long().to(accelerator.device)
        current_poi_t=torch.as_tensor(current_poi).float().to(accelerator.device)
        '''
        accelerator.print('history_data_t',history_data_t.shape)
        accelerator.print('history_dur_t',history_dur_t.shape)
        accelerator.print('history_hour_t',history_hour_t.shape)
        accelerator.print('history_day_t',history_day_t.shape)
        accelerator.print('history_poi_t',history_poi_t.shape)

        accelerator.print('current_data_t',current_data_t.shape)
        accelerator.print('current_dur_t',current_dur_t.shape)
        accelerator.print('current_hour_t',current_hour_t.shape)
        accelerator.print('current_day_t',current_day_t.shape)
        accelerator.print('current_poi_t',current_poi_t.shape)
        '''

        history_data_t=self.normalize_loc(history_data_t,'norm')
        current_data_t=self.normalize_loc(current_data_t,'norm')
        #normalize web mercator
        #accelerator.print('history_data_t+',history_data_t.shape,history_data_t[52][0])
        #accelerator.print('current_data_t+',current_data_t.shape,current_data_t[52][0])
        history_dur_t=self.normalize_dur(history_dur_t,'norm')
        current_dur_t=self.normalize_dur(current_dur_t,'norm')
        #accelerator.print('history_dur_t',history_dur_t.shape,history_dur_t[52][0])
        #accelerator.print('current_dur_t',current_dur_t.shape,current_dur_t[52][0])

        #minmax standardize duration
        #print('after norm',history_dur_t.shape)

        ################################### initial embedding and vectors##########################################
        history_loc_embedding=self.mer2vec(history_data_t)
        #batch_size,his_seq_len——>batch_size,his_seq_len,self.llm_dim
        current_loc_embedding=self.mer2vec(current_data_t)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,self.llm_dim

        #accelerator.print('history_loc_embedding',history_loc_embedding.shape,history_loc_embedding[52][0])
        #accelerator.print('current_loc_embedding',current_loc_embedding.shape,current_loc_embedding[52][0])


        history_day_embedding=self.day_embedding(history_day_t)
        #batch_size,his_seq_len——>batch_size,his_seq_len,self.llm_dim
        current_day_embedding=self.day_embedding(current_day_t)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,self.llm_dim

        #accelerator.print('history_day_embedding',history_day_embedding.shape,history_day_embedding[52][0])
        #accelerator.print('current_day_embedding',current_day_embedding.shape,current_day_embedding[52][0])

        history_hour_embedding=self.hour_embedding(history_hour_t)
        #batch_size,his_seq_len——>batch_size,his_seq_len,self.llm_dim
        current_hour_embedding=self.hour_embedding(current_hour_t)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,self.llm_dim

        #accelerator.print('history_hour_embedding',history_hour_embedding.shape,history_hour_embedding[52][0])
        #accelerator.print('current_hour_embedding',current_hour_embedding.shape,current_hour_embedding[52][0])
        
        history_dur_vector=history_dur_t.unsqueeze(-1)
        #batch_size,his_seq_len——>batch_size,his_seq_len,1
        history_dur_vector=self.dur_linear(history_dur_vector)
        #batch_size,his_seq_len,1——>batch_size,his_seq_len,self.llm_dim
        current_dur_vector=current_dur_t.unsqueeze(-1)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,
        current_dur_vector=self.dur_linear(current_dur_vector)
        #batch_size,cur_seq_len,1——>batch_size,cur_seq_len,self.llm_dim
        #accelerator.print('history_dur_vector',history_dur_vector[52][0])

        #accelerator.print('current_dur_vector',current_dur_vector[52][0])
        if(self.if_llm_poi==False):
                #accelerator.print('current_dur_vector',current_dur_vector.shape,current_dur_vector)
                history_poi_vector=self.poi_linear(history_poi_t)
                #batch_size,his_seq_len,5——>batch_size,his_seq_len,self.llm_dim
                current_poi_vector=self.poi_linear(current_poi_t)
                #batch_size,cur_seq_len,5——>batch_size,cur_seq_len,self.llm_dim
        else:
                with torch.no_grad():
                    vectors_his=self.vectors.expand(batch_num,self.his_seq_len,-1,-1,-1).to(accelerator.device)
                    #[64, 40, 5, 17, 768]
                    vectors_cur=self.vectors.expand(batch_num,self.cur_seq_len,-1,-1,-1).to(accelerator.device)
                    #print(vectors_cur.shape)
                    #[64, 5, 5, 17, 768]

                    #print(history_poi_t.shape)
                    #accelerator.print('vectors_his',vectors_his)
                    history_poi_t=history_poi_t.unsqueeze(-1).unsqueeze(-1)

                    #print(history_poi_t.shape)
                    #[64, 40, 5, 1, 1]
                    #history_poi_t= history_poi_t.expand(-1, -1, -1, 17, 768)
                    history_poi_t= history_poi_t.expand(-1, -1, -1, self.max_len_poi, self.llm_dim)
                    #accelerator.print('history_poi_t',history_poi_t)
                    #[64, 40, 5, 17, 768]
                    current_poi_t=current_poi_t.unsqueeze(-1).unsqueeze(-1)
                    #[64, 5, 5, 1, 1]
                    #current_poi_t= current_poi_t.expand(-1, -1, -1, 17, 768)
                    current_poi_t= current_poi_t.expand(-1, -1, -1, self.max_len_poi, self.llm_dim)
                    #[64, 5, 5, 17, 768]
                    #accelerator.print('current_poi_t',current_poi_t)
                    his_poi_vector=history_poi_t*vectors_his
                    #[64, 40,5, 17, 768]
                    #accelerator.print('his_poi_vector',his_poi_vector)
                    cur_poi_vector=current_poi_t*vectors_cur
                    #[64, 40,5, 17, 768]
                    #accelerator.print('cur_poi_vector',cur_poi_vector)
                    his_poi_vector=torch.sum(his_poi_vector,dim=2)
                    #[64, 40,17, 768]
                    cur_poi_vector=torch.sum(cur_poi_vector,dim=2)
                    #[64, 5,17, 768]
                    
                    his_poi_vector=his_poi_vector.reshape(batch_num,self.his_seq_len,-1)
                    #[60,40,17*768]
                    #accelerator.print('his_poi_vector',his_poi_vector)
                    cur_poi_vector=cur_poi_vector.reshape(batch_num,self.cur_seq_len,-1)   
                    #accelerator.print('cur_poi_vector',cur_poi_vector)
            #print('hist',his_poi_vector.shape)
            #print('cur',cur_poi_vector.shape)    
                #accelerator.print(1,his_poi_vector)
                his_poi_vector=self.his_additional_poi(his_poi_vector)
            #[60,40,768]
                cur_poi_vector=self.cur_additional_poi(cur_poi_vector)
        #[60,40,768]
        #accelerator.print('history_poi_vector',history_poi_vector.shape,history_poi_vector)
        #accelerator.print('current_poi_vector',current_poi_vector.shape,current_poi_vector)
        #history_all_embedding=current_loc_embedding
        #accelerator.print('his',history_loc_embedding[52][0])
        #accelerator.print('his dat',history_day_embedding[52][0])
        #accelerator.print('his hour',history_hour_embedding[52][0])
        #accelerator.print('his poi',his_poi_vector[52][0])
        #accelerator.print('cur poi',his_poi_vector[52][0])
        #accelerator.print('his dur',history_dur_vector[52][0])
        #accelerator.print(2,his_poi_vector)
        if(self.if_llm_poi==False):
                history_all_embedding=torch.concat([history_loc_embedding,
                                                history_day_embedding,
                                                history_hour_embedding,
                                                history_dur_vector,
                                                history_poi_vector],
                                                dim=-1)
                #B,L,dim
                #print('before',history_all_embedding[:,0,:])
                history_all_embedding=history_all_embedding.transpose(1,2)
                #B.dim,L
                #history_all_embedding=self.batch_norm(history_all_embedding)
                history_all_embedding=history_all_embedding.transpose(1,2)
                #B,L,dim
                #print('after',history_all_embedding[:,0,:])


                current_all_embedding=torch.concat([current_loc_embedding,
                                                current_day_embedding,
                                                current_hour_embedding,
                                                current_dur_vector,
                                                current_poi_vector],
                                                dim=-1)
                #print('before',current_all_embedding[:,0,:])
                current_all_embedding=current_all_embedding.transpose(1,2)
                #B.dim,L
                #current_all_embedding=self.batch_norm(current_all_embedding)
                current_all_embedding=current_all_embedding.transpose(1,2)
                #print('after',current_all_embedding[:,0,:])
                #print('%'*100)
        else:
                history_all_embedding=torch.concat([history_loc_embedding,
                                                history_day_embedding,
                                                history_hour_embedding,
                                                history_dur_vector],
                                                dim=-1)
                #B,L,dim
                #print('before',history_all_embedding[:,0,:])
                '''
                history_all_embedding=history_all_embedding.transpose(1,2)
                #B.dim,L
                history_all_embedding=self.batch_norm(history_all_embedding)
                history_all_embedding=history_all_embedding.transpose(1,2)
                #B,L,dim
                #print('after',history_all_embedding[:,0,:])
                '''

                current_all_embedding=torch.concat([current_loc_embedding,
                                                current_day_embedding,
                                                current_hour_embedding,
                                                current_dur_vector],
                                                dim=-1)
                #print('before',current_all_embedding[:,0,:])
                '''
                current_all_embedding=current_all_embedding.transpose(1,2)
                #B.dim,L
                current_all_embedding=self.batch_norm(current_all_embedding)
                current_all_embedding=current_all_embedding.transpose(1,2)
                #print('after',current_all_embedding[:,0,:])

                #print('%'*100)
                '''
        #########################################################################################################

        #accelerator.print('history_all_embedding',history_all_embedding.shape,history_all_embedding[52][0])
        #accelerator.print('current_all_embedding',current_all_embedding.shape,current_all_embedding[52][0])
        #print('history@@emb',history_all_embedding.shape)
        #########################################################load to llm size ########################################
        history_llm_embedding=self.history_llmize_linear(history_all_embedding)
        #accelerator.print('history_llm_embedding',history_llm_embedding.shape,history_llm_embedding[52][0])
        
        #m = torch.nn.ELU(alpha=1)
        #history_llm_embedding=m(history_llm_embedding)
        #history_llm_embedding=torch.tanh(history_llm_embedding)#+his_poi_vector
        #accelerator.print('history_llm_embedding',history_llm_embedding.shape,history_llm_embedding[52][0])
        ##batch_size,his_seq_len,llm_dim
        current_llm_embedding=self.current_llmize_linear(current_all_embedding)
        #accelerator.print('current_llm_embedding1',current_llm_embedding.shape,current_llm_embedding[52][0])
        #current_llm_embedding=torch.tanh(current_llm_embedding)#+cur_poi_vector
        #current_llm_embedding=m(current_llm_embedding)
        #accelerator.print('current_llm_embedding1',current_llm_embedding.shape,current_llm_embedding[52][0])
        ##batch_size,cur_seq_len,llm_dim
        #accelerator.print('history_llm_embedding',history_llm_embedding.shape,history_llm_embedding)
        #accelerator.print('current_llm_embedding',current_llm_embedding.shape,current_llm_embedding)
        if(self.if_llm_poi==True):
            history_llm_embedding=torch.tanh(history_llm_embedding)+his_poi_vector
            current_llm_embedding=torch.tanh(current_llm_embedding)+cur_poi_vector
        #accelerator.print('history_llm_embedding2',history_llm_embedding.shape,history_llm_embedding[52][0])
        #accelerator.print('current_llm_embedding2',current_llm_embedding.shape,current_llm_embedding[52][0])
        total_traj_embedding=torch.concat([history_llm_embedding,current_llm_embedding],dim=1)
        #total_traj_embedding=current_llm_embedding
        #batch_size,his_seq_len+cur_seq_len,llm_dim
        #accelerator.print('total_traj_embedding',total_traj_embedding.shape,total_traj_embedding)

        ############################################################################################################
        
        
       
        if self.if_prompt:
            
            user=batch['uid']
            prompts=[]
            for user_id in user:
                
                prompt=(
                    f"<|start_prompt|>Task Description: Predict the next possible location, in normalized mercator coordinates, of a resident based on their historical and current movement trajectory. "
                    f"Data Description: This dataset includes mobility trajectory data of residents. "
                    f"Each record consists of historical and current trajectories. "
                    f"The historical trajectory 40 records, while the current trajectory consists of 5 records, all sequentially arranged in chronological order. "
                    f"Additional Description: "
                    f"Historical records effectively describe the resident\’s regular travel patterns and frequently visited places, "
                    f"while current trajectories better reflect the user’s current location and their short-term travel intentions. "
                    f"Each record combines normalized mercator coordinates, day of week, time of day, duration and POI categories. <|end_prompt|>"

                )
                prompts.append(prompt)

            prompt_emb=self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(accelerator.device)
            
            prompt_embeddings = self.model.get_input_embeddings()(prompt_emb)
            #accelerator.print('prompt_embeddings',prompt_embeddings[52][0],prompt_embeddings.shape)



            total_embedding=torch.concat([prompt_embeddings,total_traj_embedding],dim=1)
        else:
            total_embedding=total_traj_embedding
        
        #total_embedding=total_traj_embedding
        #accelerator.print('total_embedding',total_embedding.shape,total_embedding[52][0])
        dec_out = self.model(inputs_embeds=total_embedding).last_hidden_state
        #batch,prompt_seq_len+his_seq_len+cur_seq_len,llm_dim
        #accelerator.print('dec_out',dec_out.shape,dec_out[52][0])
        #dec_out=dec_out#+total_embedding
        #dec_out=history_all_embedding
        return dec_out



    def predict_loc(self,next_llm_emb,accelerator):
        
        next_loc=self.output_loc(next_llm_emb)
        #batch,loc_size
        #accelerator.print('next_loc before tanh',next_loc.shape,next_loc[0])
        #accelerator.next_loc=torch.tanh(next_loc)
        #这个tanh效果其实不大
        #accelerator.print('next_loc after tanh',next_loc.shape,next_loc[0])
        return next_loc
        #batch,2

    def predict_dur(self,next_llm_emb,accelerator):
        
        next_dur=self.output_dur(next_llm_emb)
        #batch,1
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        next_dur=torch.relu(next_dur)
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        return next_dur




    def calculate_loss(self,batch,accelerator,flag):

        
        next_loc,next_loc_denorm,next_dur=self.forward(batch,accelerator)

        
        #batch,2

        lambda_dur=1

        #accelerator.print('next_loc',next_loc_denorm)
        if self.if_sim==False:

            true_loc=torch.as_tensor(batch['target']).float().to(accelerator.device)
            batch_num=true_loc.shape[0]
            #accelerator.print('true_loc',true_loc.shape,true_loc)
            #accelerator.print('tnext_loc_denorm',next_loc_denorm.shape,next_loc_denorm)
            #accelerator.print('*'*100)
            #distance_loss=self.lambda_loss*torch.dist(next_loc_denorm,true_loc)/batch_num
            distance_loss=torch.linalg.norm(next_loc_denorm- true_loc, dim=1).mean()
            #accelerator.print('dis',distance_loss,distance_loss.device)
            #accelerator.print('nex',next_loc_denorm,true_loc,true_loc.device)
            if(flag==0):
                pass
                #accelerator.print('next_loc_denorm',next_loc_denorm)
                #accelerator.print('true_loc',true_loc)
                #accelerator.print('next_loc',next_loc)
                #true_loc_norm=self.normalize_loc(true_loc,'norm')
                #accelerator.print('true_loc_norm',true_loc_norm)
                
                #accelerator.print(distance_loss)
                #accelerator.print('*'*10)
        else:
            #print('next_loc in calc',next_loc.shape,next_loc)
            
            #dur_loss=nn.CrossEntropyLoss()
            dur_loss=nn.NLLLoss()
            coordinates_tensor=torch.FloatTensor(self.coordinates).to(accelerator.device)
            #print('coordinates_tensor',coordinates_tensor,coordinates_tensor.shape)
            coordinates_tensor=self.normalize_loc(coordinates_tensor,'norm')
            #print('coordinates_tensor',coordinates_tensor,coordinates_tensor.shape)


            true_loc_emb=self.mer2vec(coordinates_tensor)
            #print('true_loc_emb',true_loc_emb,true_loc_emb.shape)
            #(N,mer_emb)


            next_loc = normalize_rows(next_loc)
            true_loc_emb = normalize_rows(true_loc_emb)

            sm=nn.Softmax(dim=1)
            sim_prob=torch.matmul(next_loc,true_loc_emb.t())
            #print('inner',sim_prob.shape,sim_prob)

            sim_prob = sm(sim_prob)
            #print('sodtmax',sim_prob.shape,sim_prob)
            #print(sim_prob[sim_prob>=0.05])

            sim_prob=torch.log(sim_prob)
            #print('log',sim_prob.shape,sim_prob)
            #batch,N
            #print('sim_prob',sim_prob,sim_prob.shape)
            #sim_prob=sim_prob/torch.sqrt(torch.tensor(self.dim_size, dtype=torch.float32))
            #print('sim_prob',sim_prob,sim_prob.shape)
            #sim_prob=torch.softmax(sim_prob,dim=-1)
            #sim_prob=torch.log(sim_prob)
            true_loc=torch.as_tensor(batch['target_idx']).long().to(accelerator.device)
            #print('true_loc',true_loc,true_loc.shape)
            '''
            accelerator.print(sim_prob,true_loc)
            accelerator.print('l',len(sim_prob[sim_prob>0.01]))
            accelerator.print('where',torch.where(sim_prob>0.01))
            accelerator.print('value',sim_prob[sim_prob>0.01])
            '''

            distance_loss=dur_loss(sim_prob,true_loc)
            #print(distance_loss)
            #accelerator.print(distance_loss)
            #accelerator.print('@'*10)
            lambda_dur=0.1
        #accelerator.print('dis',distance_loss)
        #

        #entropy_loss=cross_entropy_func(sim_prob,true_loc)

        true_dur=torch.as_tensor(batch['target_dur']).float().to(accelerator.device)
        mse_loss_func=torch.nn.MSELoss()
        mse_loss=mse_loss_func(next_dur,true_dur.view(-1, 1))

        if self.if_dur_loss:
            total_loss=distance_loss+lambda_dur*mse_loss
            #print('loss seperate',distance_loss,mse_loss,total_loss)
        else:
            total_loss=distance_loss

        return total_loss,distance_loss,mse_loss

        