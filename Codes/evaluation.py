import numpy as np
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

def multi_recall(quotas,candis,num):    
    candi_docs = []
    for i in range(len(quotas)):
        n = int(num*quotas[i])-1
        n = max(n,1)
        candi_docs += candis[i,:n].tolist()
    candi_docs = list(set(candi_docs))
    assert len(candi_docs)<=num
    inx = quotas.argmax()
    n = int(num*quotas[inx])
    for j in range(n,num):
        if candis[inx,j] in candi_docs:
            continue
        candi_docs.append(candis[inx,j])
        if len(candi_docs) == num:
            break

    candi_docs = set(candi_docs)
    assert len(candi_docs)<=num
    return candi_docs

def FedRec_Recall(lambd,clipvalue,distance,user_model,user_vecs_model,impression,test_user,news_scoring,normalized_news_scoring,Candi = [100,200,300,400,500]):
    
    click_vecs_input = Input(shape=(50,256,))
    click_mask_input = Input(shape=(50,))
    temp_user_vec = user_model.layers[-8]([click_vecs_input,click_mask_input])
    temp_user_vec = user_model.layers[-7](temp_user_vec)
    user_vec_model = Model([click_vecs_input,click_mask_input],temp_user_vec)
            
    base_key, base_value = user_model.layers[10].weights
    base_key = K.eval(base_key)
    base_value = K.eval(base_value)
    
    
    
    
    Future = {}
    History = {}
    for num in Candi:
        Future[num] = []
        History[num] = []
        
    for i in range(len(impression)):
        labels = impression[i]['labels']
        docids = impression[i]['docs']
        docids = set(docids)
        
        his_ids = set(test_user['click'][i].tolist())-{0}
        
        click = test_user['click'][i]
        click_vecs = news_scoring[click]
        click_vecs = click_vecs.reshape((1,50,256))
    
        click_num = (click>0).sum()


        context_user_vecs = user_vecs_model.predict(click_vecs)
        context_user_vecs = context_user_vecs.reshape((50,256))

        if click_num >1:
            Z = linkage(context_user_vecs[-click_num:],  method='ward', metric='euclidean')
            clusters = fcluster(Z, distance, criterion='distance')
        elif click_num == 1:
            clusters = np.array([1,])
            
        if click_num<50 :
            tmp = np.zeros((50-click_num,))
            clusters =np.array( tmp.tolist()+clusters.tolist(),dtype='int32')
       
        if click_num >0:
            cluser_num = clusters.max()

        context_user_vecs = context_user_vecs.reshape((1,50,256))
        

        if (click>0).sum()==0:
            uvs = np.zeros((1,256))
            quotas = np.array((1,))
        else:        
            uvs = []
            quotas = []
            for j in range(1,cluser_num+1):
                mask = np.array(clusters==j,dtype='float32')
                mask = mask.reshape((1,50))
                user_vec = user_model.layers[9].predict([context_user_vecs,mask])
                user_vec = user_vec.reshape((256,))
                
                att_score = np.dot(user_vec,base_key)
                if lambd>0:
                    att_score[att_score>clipvalue] = clipvalue
                    att_score[att_score<-clipvalue] = -clipvalue
                    noise = np.random.laplace(0,lambd,size=(30,))
                    att_score += noise
                att_score = np.exp(att_score)
                att = att_score/att_score.sum()
                user_vec = np.dot(att,base_value)
     
                
                
                uvs.append(user_vec)
                quotas.append(mask.sum()/(click>0).sum())

            uvs = np.array(uvs)
            quotas = np.array(quotas)


        
        num = Candi[-1]
        scores = np.dot(uvs,normalized_news_scoring.T)
        argsort = (-scores).argsort(axis=-1)
        candis = argsort[:,:num]
        
        for num in Candi:
            candi_docs = multi_recall(quotas,candis,num)
            r = len(set(docids&candi_docs))/len(docids)
            Future[num].append(r*100)
            if len(his_ids)>0:
                r = len(set(his_ids&candi_docs))/len(his_ids)
                History[num].append(r*100)
        if i%2000==0:
            gf = {}
            gh = {}
            for num in Future:
                gf[num] = np.array(Future[num]).mean(axis=-1)
                gh[num] = np.array(History[num]).mean(axis=-1)
            print(gf)
            print(gh)
            print()
        
    for num in Candi:
        Future[num] = np.array(Future[num]).mean(axis=-1)
        History[num] = np.array(History[num]).mean(axis=-1)

        
    return Future, History