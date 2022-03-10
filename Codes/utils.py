import random
import numpy as np

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

def shuffle(pn,labeler,pos):
    index=np.arange(pn.shape[0])
    pn=pn[index]
    labeler=labeler[index]
    pos=pos[index]
    
    for i in range(pn.shape[0]):
        index=np.arange(npratio+1)
        pn[i,:]=pn[i,index]
        labeler[i,:]=labeler[i,index]
    return pn,labeler,pos

def Log(loss,acc,per,K=2):
    NUM = 100//K
    num1 = int(NUM*per)
    num2 = NUM-num1

    print("\r loss=%.3f acc=%.4f  %s%s  %.2f%s" % (loss,acc,'>' * num1,'#'*num2, 100*per,'%'), flush=True, end='')

def convert(L1,budget,user_num):
    lambd = L1/budget
    sigma = np.sqrt(2/user_num)*lambd
    return lambd,sigma