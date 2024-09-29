import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from libcity.model.abstract_model import AbstractModel


class FPMC(AbstractModel):
    """
    """

    def __init__(self, config, data_feature):
        super(FPMC, self).__init__(config, data_feature)
        print(data_feature)
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.uid_size = data_feature['uid_size']
        self.loc_size = data_feature['loc_size']

        # ??? FPMC ???????? Embedding
        self.UI_emb = nn.Embedding(self.uid_size, self.embedding_size)
        self.IU_emb = nn.Embedding(
            self.loc_size, self.embedding_size,
            padding_idx=data_feature['loc_pad'])
        self.LI_emb = nn.Embedding(
            self.loc_size, self.embedding_size,
            padding_idx=data_feature['loc_pad'])
        self.IL_emb = nn.Embedding(
            self.loc_size, self.embedding_size,
            padding_idx=data_feature['loc_pad'])

       

    def forward(self, batch):
        # Embedding ????

        last_loc_index = torch.LongTensor(batch.get_origin_len(
            'current_loc')) - 1  # Markov chain ????????????,???????????
        last_loc_index = last_loc_index.to(self.device)
        # batch_size * 1
        last_loc = torch.gather(
            batch['current_loc'], dim=1, index=last_loc_index.unsqueeze(1))

        user_emb = self.UI_emb(batch['uid'])  # batch_size * embedding_size
        last_loc_emb = self.LI_emb(last_loc)  # batch_size * 1 * embedding_size

        all_iu_emb = self.IU_emb.weight  # loc_size * embedding_size
        mf = torch.matmul(user_emb, all_iu_emb.transpose(0, 1))

        all_il_emb = self.IL_emb.weight
        fmc = torch.matmul(last_loc_emb, all_il_emb.transpose(0, 1))
        fmc = torch.squeeze(fmc, dim=1)
        score = mf + fmc  # batch_size * loc_size
        return score

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        # ?? loss ???,????
        criterion = nn.NLLLoss().to(self.device)
        scores = self.forward(batch)
        return criterion(scores, batch['target'])
