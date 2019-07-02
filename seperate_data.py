import pandas as pd
import numpy as np
import os
import threading as td
from queue import Queue
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.utils.data as Data
import xlsxwriter
import time
from sklearn.preprocessing import OneHotEncoder
COLSIZ = 10
tformat = lambda s: str(s).title().ljust(COLSIZ)

def spart_and_one_hot_enc(url=None,store_url=None,fname=None):
    """
    func: separate data to Attack And Normal
    :param url: dataset file dir
    :return:None
    """
    data1 = pd.read_pickle(url,compression='zip')
    one_hot_enc = OneHotEncoder(categories=[range(16), range(16), range(16)])

    data = data1.values.astype(np.float64)
    print('{} has data shaped:{}'.format(fname, data.shape))
    rows = data.shape[0]
    start = 0
    row = int(rows // 64)
    end = row*64
    source_flags = data[start:end,-1].tolist()
    a_count = 0
    n_count = 0
    atta_url = os.path.join(store_url,'{}_pure_attack.csv'.format(fname))
    norl_url = os.path.join(store_url,'{}_pure_normal.csv'.format(fname))
    print('{}'.format(' '.join(map(tformat, ('attack', 'row', 'a_count', 'n_count')))))

    for r in range(row):
        num = 0
        if r % 2000 == 0:
            print('{}'.format(' '.join(map(tformat,(fname,row,a_count,n_count)))))
        if 1. in source_flags[r*64:r*64+64]:# or 1 in source_flags[r*64:r*64+64]:
            num = 1
        dat = one_hot_enc.fit_transform(data[r * 64:r * 64 + 64, 1:4]).toarray()
        label = pd.DataFrame(source_flags[r * 64:r * 64 + 64])

        if num:
            # attack data
            # test point
            atta = pd.DataFrame(dat)

            atta = pd.concat((atta,label),axis=1)
            atta.to_csv(atta_url,sep=',',header=False,index=False,columns=None,mode='a',index_label=None,encoding='utf-8')
            a_count += 1
        else:
            # Normal data
            nor = pd.DataFrame(dat)

            nor = pd.concat((nor,label),axis=1)
            nor.to_csv(norl_url,sep=',',header=False,index=False,columns=None,mode='a',index_label=None,encoding='utf-8')
            n_count += 1

    print('{},from {} to {} acquires {} blocks,labels lengh attack|normal :{}|{},done!!!\n'.
          format(fname, start,end,row,a_count,n_count))
    return row, a_count,n_count


if __name__ == '__main__':
    print('start at:{}'.format(time.asctime(time.localtime(time.time()))))
    test_addr = '/home/yyd/dataset/hacking/ignore_ID_diff_time'
    res_addr = '/home/yyd/dataset/hacking/one-hot-repeat-lab'
    for i in os.listdir(test_addr):
        fname = i.split('_')[0]
        store_url = os.path.join(res_addr, fname)
        if not os.path.exists(store_url):
            os.makedirs(store_url)
        row, a_count,n_count = spart_and_one_hot_enc(os.path.join(test_addr,i),store_url,fname)
        # break

        for ii in os.listdir(store_url):
            if 'csv' in ii:
                pass
            else:
                continue
            ii = os.path.join(store_url,ii)
            j = os.path.splitext(ii)[0]+'.pkl'
            pd.read_csv(ii,sep=None,delimiter=',',dtype=np.float64,header=None,engine='python',encoding='utf-8').\
                to_pickle(j,compression='zip')
        print('\n')
        # break
    print('end at:{}'.format(time.asctime(time.localtime(time.time()))))
