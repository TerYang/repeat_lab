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
from torch.autograd import Variable

title = lambda string: string.title()

BATCH_SIZE = 64

# test_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/new_data/"
#
# source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/"

# 大数据下训练:验证:测试比例 98:1:1
def DataloadtoGAN(path,mark=None,label=False,single_dataset=False,hacking=False,select=''):
    """
    func: read normal data to train GAN defined as the Class(new gan code)
    :param path:dataset url
    :param mark:'validate' 'test' 'train'
    :param label: whether deliver label
    :param single_dataset:  for whole normal dataset,not suitable for normal status dataset from hacking dataset
    :param hacking: whole normal dataset or  normal status dataset from hacking dataset
    :return: dataloader,torch.Tensor

    """
    if mark == None:
        print('mark is None, please checks')
        return
    if hacking:
        files = []
        for d in os.listdir(path):
            if select.title() == d.title():
                if mark=='validate':
                    f = os.path.join(path, d, 'pure_normal.pkl')
                else:
                    f = os.path.join(path,d,'pure_attack.pkl')
                files.append(f)
                # print(files[0])
                break
            elif select=='':
                pass
            else:
                continue

            if 'normal.pkl' in d:
                files.append(os.path.join(path, d))
                continue
            elif '.' in d:
                continue
            else:
                for f in os.listdir(os.path.join(path, d)):
                    if 'normal.pkl' in f:
                        files.append(os.path.join(path,d, f))
    else:
        files = [os.path.join(path,f) for f in os.listdir(path) if 'pkl' in f]

    fl = []
    if single_dataset:
        files = [i for i in files if 'Attack_free_dataset2' in i]
    data2 = np.empty((64,21))
    atta2 = np.empty((64,21))

    # read dataset
    for i,f in enumerate(files):
        print('address:%s'%f)
        atta = np.empty(((64,21)))

        data1 = pd.read_pickle(f,compression='zip')
        data = data1.values.astype(np.float64)
        # file = os.path.basename(path)
        rows = data.shape[0]
        start = 0
        end = rows
        row = int(rows // 64)
        row1 = row
        file = os.path.splitext(os.path.basename(f))[0]
        fl.append(file)
        dirname = os.path.dirname(f).split('/')[-1]

        if mark == 'test':
            start = int(((rows*0.99)//64)*64)
            row = int((rows*0.01)//64)
            if start % 64 == 0:
                pass
            else:
                start = ((start // 64) + 1) * 64
            # end = int(start+((rows-start)//64)*64)
            end = int(start+row*64)
        elif mark == 'train':
            print('get type:%s'%'train')
            # row = int((rows*0.01)//64)
            # end = int(row * 64)
            row = int((rows*0.98)//64)
            end = int(row * 64)
        elif mark == 'validate':
            print('get type:%s,datatype:%s'%('validate',dirname))
            row = int((rows*0.01)//64)
            start = int(((rows*0.98)//64) * 64)
            if start % 64 == 0:
                pass
            else:
                start = int(((start // 64) + 1) * 64)
            end = int(start + row*64)
            # end =int(((rows*0.99)//64) * 64)

        if hacking:
            data = data[start:end, :-1].reshape((-1, 21))
            if mark == 'validate' or mark == 'test':
                url = os.path.dirname(f) + '/pure_attack.pkl'
                atta = pd.read_pickle(url,compression='zip')
                atta = pd.DataFrame(atta).to_numpy().reshape((-1,64,22))
                print('{},shape:{}'.format('pure_attack',atta.shape),end=',')
                print('start at:{},%64={},end:{},%64={},acquires row:{},percent:{}%,done read files!!!'.
                      format(start, start%64, end, end%64, row,float(row/atta.shape[0])))
                atta = atta[:row,:,:21]
                # print('atta.shape---:',atta.shape)

            if i > 0:
                data2 = np.concatenate((data2, data), axis=0).reshape((-1, 21))
                atta2 = np.concatenate((atta2, atta), axis=0)
                # print('atta2.shape:',atta2.shape)
            else:
                data2 = data
                atta2 = atta
        else:
            data = data[start:end, :].reshape((-1, 21))
            if i > 0:
                data2 = np.concatenate((data2,data),axis=0).reshape((-1,21))
            else:
                data2 = data
        print('{} shaped:{},trunked:{}'.format(file, data1.shape, data.shape),end=',')
        print('get|all:{}|{},blocks:{}'.format(row, row1, row % 64),end=',')
        print('start at:{},%64={},end:{},%64={},percent:{}%,done read files!!!'.
              format(start,start%64, end, end%64,float(row/row1)))
        # exit()
    if mark == 'validate' or mark=='test':
        atta2 = atta2.reshape((-1,64,21))
        data2 = data2.reshape((-1, 64, 21))
        label1 = np.ones((atta2.shape[0],1))
        label0 = np.zeros((data2.shape[0],1))

        data2 = np.concatenate((data2,atta2),axis=0)
        labels = np.concatenate((label0,label1),axis=0)

        TraindataM = torch.from_numpy(data2).float()  # transform to float torchTensor
        TraindataM = torch.unsqueeze(TraindataM, 1)
        Traindata_LabelM = torch.from_numpy(labels).float()
        TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)

        print('{},size:{} label:{},done read files!!!\n'.format('validate mix dataset', TraindataM.shape, label))
        return Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    TraindataM = torch.from_numpy(data2.reshape((-1, 64, 21))).float()    # transform to float torchTensor
    TraindataM = torch.unsqueeze(TraindataM,1)

    if label:
        # if mark == 'train' or mark == 'test':
        if select == 'Normal':
            labels = np.zeros((TraindataM.shape[0],1))
        else:
            labels = np.ones((TraindataM.shape[0], 1))

        Traindata_LabelM = torch.from_numpy(labels).float()
        TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
        print('{},size:{} label:{},done read files!!!\n'.format(fl, TraindataM.shape, label))
        return Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        # if mark == 'train' or mark == 'test':
        # Data Loader for easy mini-batch return in training
        TorchDataset = Data.TensorDataset(TraindataM)
        print('{},size:{} label:{},done read files!!!\n'.format(fl, TraindataM.shape, label))
        return Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)


def base_read(path,mark=str,target_type=str,bias_dataset=str):
    """
    func: base read file func
    :param path:the basedir of kind of attack type ,containing dataset files
    :param mark: read file for what
    :param target_type: the read target of type of dataset file
    :param bias_dataset: whether read dataset such as either normal data or attack data, or both of them
    :return:  list of list of label,list of 3 dim array of data,dataset name
    """
    urls = []
    file = path.split('/')[-1]
    fg = 0
    print('get type:%s' % mark,end=',')
    for i in os.listdir(path):
        if bias_dataset.title() == 'Both':
            if 'normal'.lower() in i and target_type in i:
                fg += 1
                urls.append(os.path.join(path,i))

            elif 'attack'.lower() in i and target_type in i:
                fg += 1
                urls.append(os.path.join(path,i))
        else:
            if bias_dataset.lower() in i and target_type in i:
                urls.append(os.path.join(path,i))
                fg += 1
    if fg != 2 and bias_dataset.title() == 'Both':
        print('base read function arise error,because {} has not normal or attack {} file to read'.format(file, target_type))
    elif fg != 1 and bias_dataset.title() != 'Both':
        print('base read function arise error,because {} has not {} {} type file to read'.
              format(file,bias_dataset,target_type))
    flags = []
    data_list = []
    # print('-'*20,file,'-'*20)
    for j, i in enumerate(urls):
        flag = []
        if target_type == 'pkl':
            try:
                data1 = pd.read_pickle(i,compression='zip')
            except:
                print('read file error at pd read_pickle!')
                return
        elif target_type == 'csv':
            try:
                data1 = pd.read_csv(i, sep=',', header=None, dtype=np.float64, engine='python', encoding='utf-8')#,nrows=64*64*100 ,nrows=64*64*1
            except:
                print('read file error at pd read_csv!')
                return
        elif target_type == 'txt':
            try:
                data1 = pd.read_csv(i, sep=None, header=None, dtype=np.float64, engine='python',
                                encoding='utf-8')  # ,nrows=64*64*100 ,nrows=64*64*1
            except:
                print('read file error at pd read txt file!')
                return
        else:
            print('param %s is illegal!!!'%target_type)
            return
        filename = os.path.basename(i)
        data = data1.values.astype(np.float64)

        rows = data.shape[0]
        column = data.shape[1]
        print('data column:',column)
        start = 0
        row = int(rows // 64)
        end = int(row*64)
        if mark:
            if mark == 'test':
                start = int(int(row * 0.81)*64)
                row = int(row * 0.19)
                end = int(start + row * 64)
            elif mark == 'train':
                row = int(row * 0.8)
                end = int(row * 64)
            elif mark == 'validate':
                row = int(row * 0.01)
                start = int(int((row * 0.8)) * 64)
                end = int(start + row * 64)
                print('row:{},row%64={}|{}'.format(row, row % 64, (end - start) % 64))

            elif mark == 'coding':
                row = int(row * 0.001)
                end = int(row * 64)
        else:
            print('It is illegal that mark is None!!!')

        source_flags = data[start:end,-1].tolist()

        # batch label,if any label 1 in a batch size of data,batch label marked as 1
        count_1 = 0
        count_0 = 0
        if 'pure' in filename:
            if 'attack' in filename:
                flags.append(np.ones((row,1)).tolist())
                count_1 = row
            elif 'normal' in filename:
                flags.append(np.zeros((row,1)).tolist())
                count_0 = row
            print('{} {},{} has shape {},read by pandas,'.format(file,bias_dataset,filename,data1.shape),'label 1|0:{}|{}'.format(count_1,count_0),end=',')

        else:
            for r in range(row):
                num = 0
                if 1. in source_flags[start + r*64: start + r*64+64] or 1 in source_flags[start + r*64: start + r*64+64]:
                    num = 1
                    count_1 += 1
                flag.append(num)
            print('{} {},{} has shape {},read by pandas,'.format(file,bias_dataset,filename,data1.shape),'label 1|0 : {}|{}'.format(count_1,row-count_1),end=',')
            flags.append(flag)
        data = data[start:end,:-1].reshape((-1,64,column-1))
        data_list.append(data)
        print('{} start at:{},end at:{}, len of labels : {} data size:{},'
              'row:{} done read files!!!\n'.format(file, start,end,len(flags[-1]),data.shape,row))
    print('-'*20,'%s end!'%file,'-'*20)
    return flags,data_list,file,data_list[0].shape[-1]


def read_dataset(root_path_=str,target_type=str,read_target=str,usage=str,res_num=int,res_type='dataloader', selected=None,bias_dataset=str):#,label=True
    """
    func:read dataset to nets,get data to Nets,satisfied multifunction
    :param root_path_:basedir of dataset
    :param target_type: 'csv','pkl','txt'
    :param read_target:'all','select'
    :param usage: 'train','validate','test','coding'
    :param res_num:refine how many result return for call
    :param label: default dataloarder with label
    :param res_type: default 'dataloader'
    :param selected:if read_target== select,select the selected to read,could be 'Dos Fuzzy,RPM,gear' dataset file name
    :param bias_dataset: whether a requirement of dataset is only normal or attack,it could be 'Normal','Attack','Both'
    :return: list of dataloarder or single dataloarder contained all read dataset
    """
    print('-----------------------------------%s,%s-----------------------------'%(read_dataset.__name__,usage))
    print('data address:{}, sub-dataset:{}'.format(root_path_,os.listdir(root_path_)))
    # selected attack type to read data
    if selected != None:
        selected = list(map(title,selected))
    if read_target == 'all':
        files = [os.path.join(root_path_,f) for f in os.listdir(root_path_)]
    elif read_target == 'select':
        files = [os.path.join(root_path_,f) for f in os.listdir(root_path_) if f.title() in selected]
    else:
        print('func read_dataset: arise error at param read_target')

    # dataset_urls = []
    results = []
    for file in files:
        flag = 0
        try:
            for i in os.listdir(file):
                if target_type in i:
                    flag += 1
                    pass
            if flag==0:
                print('{} has not {} file'.format(file,target_type),'please check the file folder')
                print(os.listdir(file))
                # dataset_urls.append(os.path.join(file,i))
        except:
            print(files,'\n error for target file folder')
            return
    pool = mp.Pool(processes=len(files))

    for i in files:
        # print(i)
        # results.append(pool.apply(testdata,(os.path.join(path,i),mark,)))#_async
        results.append(pool.apply_async(base_read, (i, usage,target_type,bias_dataset,)))  # _async

    pool.close()
    pool.join()

    names = []
    flags = []
    row = 0

    column = results[0].get()[3]
    # print('column:',column)
    if res_type == 'seperate' or res_num > 1:
        data = []
    else:
        data = np.empty((64,column))

    f2 = lambda x:len(x)
    # ll = 0
    for i, result in enumerate(results):
        # result = result#.get()
        result = result.get()
        # print('i:', i,'file:',result[2])
        # # print('%s,%d'%(result[2],len(flg)))
        # flags.append(result[0])
        # data.append(result[1])
        label_ = []
        # ll = 0
        for flg in result[0]:
            # print(flg.__class__,len(flg))
            row += len(flg)

            if res_type == 'seperate':
                flags.append(flg)
            else:
                label_.extend(flg)
                #  older codes
                # if ll == 0 and i == 0:
                #     flags = flg
                #     # print('fg:',flg,flg.__class__)
                #     ll+=1
                # else:
                #     # print('flags:',flags)
                #     flags.extend(flg)
                # # print('%s,%d'%(result[2],len(flg)))
        # concat normal status and attack status or single data suche as normal status or attack status
        # of all types of attack to one container
        # concat all to one container
        if res_num == 1:
            flags.extend(label_)
        # concat to res_num containers,res_num default equal to the number of target_type
        else:
            flags.append(label_)

        la = 0
        dat = np.empty((1,64,column))
        for dt in result[1]:
            if res_type == 'seperate':
                data.append(np.array(dt).astype(np.float64).reshape((-1,64,column)))
            else:
                dt = np.array(dt).reshape((-1,64,column)).astype(np.float64)
                if la == 0:
                    dat = dt
                    la += 1
                else:
                    dat = np.concatenate((dat,dt),axis=0).reshape((-1,64,column))
                # older codes
                # if la == 0 and i == 0:
                #     data = dt
                #     la += 1
                # else:
                #     data = np.concatenate((data,dt)).reshape((-1,64,column))
        # concat normal status and attack status or single data suche as normal status or attack status
        # of all types of attack to one container
        # concat all to one container
        if res_num == 1:
            if i == 0:
                data = dat
            else:
                data = np.concatenate((data,dat)).reshape((-1,64,column))
        # concat to res_num containers,res_num default equal to the number of target_type
        else:
            data.append(dat)
        names.append(result[2])
        # row += sum(list(map(f2,result[0])))
    print('-'*20,'total result','-'*20)
    print('\n return {} blocks of data,{} blocks of label,all {} blocks'.format(data.__len__(), len(flags), row),end=',')

    if res_type == 'seperate':
        return data,flags,names
    print(names,end=',')
    if res_num == 1:
        # data_array = np.array(data).reshape((-1,64.21))
        data_array = data
        labels = np.array(flags).reshape((-1,1)).astype(np.float64)
        TraindataM = torch.from_numpy(data_array).float()  # transform to float torchTensor
        TraindataM = torch.unsqueeze(TraindataM, 1)
        Traindata_LabelM = torch.from_numpy(labels).float()
        TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
        dataloader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)
        print('return one dataloarder with {} tensors,data shape:{},label shape:{}'.
              format(len(dataloader.dataset.tensors), dataloader.dataset.tensors[0].size(),dataloader.dataset.tensors[1].size()))
        print('------------------------------------------------------------------')
        return dataloader,names
    else:
        print('result len:',len(data),len(flags))
        dataloaders = []
        f1 = lambda x: x.dataset.tensors[0].size()
        # for i,label,dat in enumerate(list(zip(flags,data))):
        for label,dat in list(zip(flags,data)):
            dat = np.array(dat).reshape((-1,64,column))
            label = np.array(label).reshape((-1,1))
            TraindataM = torch.from_numpy(dat).float()  # transform to float torchTensor
            TraindataM = torch.unsqueeze(TraindataM, 1)
            Traindata_LabelM = torch.from_numpy(label).float()
            TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
            dataloaders.append(Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True))

        print('return list of dataloader has {} dataloarders,data shape respectively:{}'.
              format(len(dataloaders), list(map(f1,dataloaders))))
        print('------------------------------------------------------------------')
        return dataloaders,names


if __name__ == '__main__':
    addr = '/home/yyd/dataset/hacking/one-hot-repeat-lab'#attack data
    # for i in os.listdir(addr):
    #     print(os.path.join(addr,i))
    # def read_dataset(root_path_=str, target_type=str, read_target=str, usage=str, res_num=int, res_type='dataloarder', selected=None, bias_dataset=str):  # ,label=True

    dataloaders,names = read_dataset(root_path_=addr,target_type='pkl',read_target='all',usage='coding',res_num=1,res_type='dataloader',bias_dataset='normal')

    print(names)
    print(dataloaders.dataset.__len__(),dataloaders.__class__,len(i.dataset.tensors))

    # flags,data,names = read_dataset(root_path_=addr,target_type='pkl',read_target='select',usage='coding',res_num=1,res_type='seperate',bias_dataset='both',selected=['rpm','gear'])
    # print(dataloaders.dataset.__len__(),dataloaders.__class__,len(dataloaders.dataset.tensors))

    # for i in dataloaders:
    #     print(i.dataset.__len__(),i.__class__,len(i.dataset.tensors))

    # flags,data,name = base_read(os.path.join(addr,'DoS'),mark='coding',target_type='pkl',bias_dataset='both')

    # print(a.dataset.__len__())
    # print(len(a.dataset.tensors))
    # print(a.dataset.tensors[1].numpy()[0])