import numpy as np
from generator import *
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

def FedTrain_single(user_num,lambd,cluster_distance,all_models,training_data):
    
    user_vecs_model,model = all_models
    news_scoring,train_user_dict,train_user,train_user_id,train_sess,train_label,bz_num = training_data
    
    fed_user_index = np.random.choice(len(train_user_dict),(user_num,),replace=False)
    fed_index = np.concatenate([train_user_dict[uid] for uid in fed_user_index])
    fed_train_generator = get_hir_train_generator(news_scoring,train_user['click'],train_user_id[fed_index],train_sess[fed_index],train_label[fed_index],bz_num)

    
    all_weights = []
    last_weights = model.get_weights()

    for datax,datay in fed_train_generator:

        title_vecs,click_vecs,single_mask = datax
        context_user_vecs = user_vecs_model.predict(click_vecs)

        masks = []
        quotas = []
        for i in range(click_vecs.shape[0]):
            Z = linkage(context_user_vecs[i],  method='ward', metric='euclidean')
            clusters = fcluster(Z, cluster_distance, criterion='distance')
            clusters_num = clusters.max()
            mask = np.zeros((MAX_CHANNEL_NUM,50))+1
            quota = np.zeros((1+npratio,MAX_CHANNEL_NUM))
            sss = single_mask[i].sum() + 10**(-6)
            if clusters_num>MAX_CHANNEL_NUM:
                mask[0,:] = single_mask[i]
                quota[:,0] = 1
            else:
                for j in range(1,1+clusters_num):
                    flag = clusters==j
                    flag = flag*single_mask[i]
                    t = np.array(flag,dtype='int32')
                    mask[j-1,:]  = t
                    quota[:,j-1] = flag.sum()/sss
            masks.append(mask)
            quotas.append(quota)
        masks = np.array(masks)
        quotas = np.array(quotas)

        datax = [title_vecs,click_vecs,masks,quotas]

        his = model.fit(datax,datay,batch_size=len(datay[0]),verbose=False)
        now_weights = model.get_weights()
        all_weights.append(now_weights)
        model.set_weights(last_weights)
    
    if lambd >0:
        now_weights = [np.average(weights, axis=0)+ np.random.laplace(0,lambd,size=[user_num]+list(weights[0].shape)).mean(axis=0) for weights in zip(*all_weights)]
    else:
        now_weights = [np.average(weights, axis=0) for weights in zip(*all_weights)]
    model.set_weights(now_weights)