from hypers import *
from utils import *
import numpy as np
import os
import pickle

def read_news(path,filenames):
    news={}
    category=[]
    subcategory=[]
    news_index={}
    index=1
    word_dict={}
    word_index=1
    with open(os.path.join(path,filenames)) as f:
        lines=f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id= splited[0]
        news_index[doc_id]=index
        index+=1
    return news_index

def read_train_clickhistory(news_index,data_root_path,filename):
    lines = []
    userids = {}
    uindex = 0
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
    # lines = []
    # with open(os.path.join(data_root_path,filename)) as f:
    #     for i in range(1000):
    #         line = f.readline()  
    #         lines.append(line) 

    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        if not uid in userids:   
            userids[uid]= uindex
            uindex += 1
        uid = userids[uid]
        sessions.append([uid,true_click,pos,neg])
    return sessions

def read_test_clickhistory(news_index,data_root_path,filename):
    lines = []
    userids = []
    # with open(os.path.join(data_root_path,filename)) as f:
    #     lines = f.readlines()
    lines = []
    with open(os.path.join(data_root_path,filename)) as f:
        for i in range(10000):
            line = f.readline()  
            lines.append(line) 
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([None,true_click,pos,neg])
    return sessions


def parse_user(news_index,session):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        _, click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user


def get_train_input(news_index,session):
    sess_pos = []
    sess_neg = []
    user_id = []
    true_user_ids = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        uid,_, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)
            true_user_ids.append(uid)

    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    true_user_ids = np.array(true_user_ids, dtype='int32')
    return sess_all, user_id, label, true_user_ids

def get_test_input(news_index,session):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _,_, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,

def load_rank_model_news_emb(news_index,data_root_path,):
    news_scoring = np.zeros((len(news_index)+1,256))
    with open(os.path.join(data_root_path,'mind_large_news_scoring_dict.pickle'),'rb') as f:
        news_scoring_dict = pickle.load(f)
    for nid, nix in news_index.items():
        news_scoring[nix] = news_scoring_dict[nid]    
    normalized_news_scoring = news_scoring/np.sqrt((news_scoring**2).sum(axis=-1).reshape((len(news_scoring),1))+10**(-8))
    
    return news_scoring,normalized_news_scoring

def get_client_data_dict(train_true_user_ids):
    train_user_dict = {}
    for i in range(len(train_true_user_ids)):
        uid = train_true_user_ids[i]
        if not uid in train_user_dict:
            train_user_dict[uid] = []
        train_user_dict[uid].append(i)
    for uid in train_user_dict:
        train_user_dict[uid] = np.array(train_user_dict[uid],dtype='int32')
        
    return train_user_dict