from keras.utils import Sequence
import numpy as np
from hypers import *

class get_hir_train_generator(Sequence):
    def __init__(self,news_info,clicked_news,user_id, news_id, label, batch_size):
        self.news_info = news_info
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __get_news(self,docids):
        news_info = self.news_info[docids]

        return news_info
        

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed]
        
        doc_ids = self.doc_id[start:ed]
        title= self.__get_news(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_title = self.__get_news(clicked_ids)
        
        mask = clicked_ids>0
        mask = np.array(mask,dtype='float32')
                

        return ([title, user_title,mask],[label])