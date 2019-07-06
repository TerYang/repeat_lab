import sys
import os
# sys.path.append('../')
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
from read_func import *
import multiprocessing
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve,roc_auc_score
import torch.nn as nn
GPU_ENBLE = True
# DATA_TYPE = 'attack_free'#'hacking_datasets'
# TEST_Gen = False # test Generator generates data at the same time

columns_ = ['pre','N_pre','F1','acc','recall',]
BATCH_SIZE = 64
Z_SIZE = 256 # generative source size
def writelog(content,url=None):
    # a = '/home/gjj/PycharmProjects/ADA/TorchGAN-your-mind/Nets/full/2019-04-17/test_logs/'
    if url == None:
        print(content)
    else:
        collect_url = './'
        if not os.path.exists(collect_url):
            os.makedirs(collect_url)
        url = os.path.join(collect_url,'{}_TestLogs.txt'.format(url))
        with open(url, 'a', encoding='utf-8') as f:
            f.writelines(content + '\n')
            print(content)


def test_attfree(path, logmark, file, test,name=None):#flags,
    """
    func: test data runs at every pkl(module)
    :param path: pkl(module) path
    :param logmark: module name
    :param test: test data, torch dataloarder
    :return:no
    #Label 0 means normal,size 1*BATCH
    # Label 1 means anormal,size 1*BATCH
    """

    print('module:',name,end=',')
    t1 = time.time()
    print('dataset:',file)
    """处理"""
    # modulename = ''
    # result = np.empty((2, 1))
    # if len(jf):
    # global Dnet
    if GPU_ENBLE:
        Dnet = torch.load(path)

    else:
        # Dnet = Dnet.cpu()
        Dnet = torch.load(path, map_location='cpu')

    TP = 0  # 1 -> 1 true positive
    TN = 0  # 0 -> 0 true negative
    FN = 0  # 1 -> 0 false negative
    FP = 0  # 0 -> 1 false positive
    import math
    def f1(l,r):
        if l == 0:
            if math.fabs(l - r) < 0.5:
                # TN
                return 0
            else:
                # FP
                return 3
        else:
            if math.fabs(l - r) < 0.5:
                # TP
                return 1
            else:
                # FN
                return 2
    total = 0
    # detail_url = './detail/{}'.format(file)
    detail_url = './{}_detail'.format(file)
    # pic_url = './{}_picture'.format(file)
    if not os.path.exists(detail_url):
        os.makedirs(detail_url)
        # os.makedirs(pic_url)
    url_numpy = os.path.join(detail_url,'{}_test_at_{}.csv'.format(logmark,file))
    y_true = []
    y_pre_ = []

    label0 = 0
    label1 = 0
    for iter, (x_, label_) in enumerate(test):
        if iter == test.dataset.__len__() // 64:
            total = iter
            break
        if GPU_ENBLE:
            flag = 1
            x_ = x_.cuda()
            try:
                Results = Dnet(x_)
            except:
                try:
                    Results, _ = Dnet(x_)
                except:
                    flag = 0
                    print('path:', path,'file:',file)
            if flag:
                pass
            else:
                return
            if Results.cpu().size(1) == 2:
                linear = nn.Linear(2, 1).cuda()
                sigmod = nn.Sigmoid().cuda()
                Results = sigmod(linear(Results))
            result = Results.data.cpu().numpy()

        else:
            flag = 1
            try:
                Results = Dnet(x_)
                # result = Results.data.numpy()
                # print(len(result))
            except:
                try:
                    Results, _ = Dnet(x_)
                    # result = Results.data.numpy()
                except:
                    flag = 0
                    print('path:', path,'file:',file)
            if flag:
                pass
            else:
                return
            if Results.size(1) == 2:
                linear = nn.Linear(2, 1)
                sigmod = nn.Sigmoid()
                Results = sigmod(linear(Results))

            result = Results.data.numpy()

        result = np.squeeze(result).tolist()
        label = np.squeeze(label_.data.numpy()).tolist()
        y_pre_.extend(result)
        y_true.extend(label)
        label0 += label.count(0)
        label1 += label.count(1)

        ll = list(map(f1, label, result))
        TN += ll.count(0)
        TP += ll.count(1)
        FN += ll.count(2)
        FP += ll.count(3)
        dat1 = pd.DataFrame({'res':result,'label':label})
        if iter:
            dat1.to_csv(url_numpy,sep=',',float_format='%.2f',header=None,index=None,mode='a',encoding='utf-8')
        else:
            dat1.to_csv(url_numpy,sep=',',float_format='%.2f',header=True,index=None,mode='a',encoding='utf-8')

    fpr,tpr = [],[]
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pre_)
    except :
        pass
    # if int(logmark)%10 == 0:
    #      fpr,tpr,_ = roc_curve(y_test,y_pre_)
         # plt.figure(1)
         # plt.plot([0, 1], [0, 1], 'k--')
         # imname = '{}'.format(logmark)
         # plt.plot(fpr, tpr,label=imname)
         # # plt.xlabel('False positive rate')
         # # plt.ylabel('True positive rate')
         # # plt.title('ROC curve')
         #
         # # result_dir = os.path.join(pic_url,'{}_test_at_{}_roc_curve.png'.format(logmark,file))
         # # plt.legend(loc='best')
         # # plt.savefig(result_dir)
         # # plt.show()
         # # plt.close()
    # release gpu memory
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    res = {}
    # 1 precision of position
    try:
        # res['pre']='{}'.format(TP/(FP+TP))
        res['pre']=TP/(FP+TP)
    except ZeroDivisionError:
        res['pre'] = 'NA'
    # 2  precision of negative
    try:
        res['N_pre']=TN/(TN+FN)
        # res['N_pre']='{}'.format(TN/(TN+FN))
    except ZeroDivisionError:
        # writelog('have no P(normaly event)',file)
        res['N_pre'] = 'NA'
    # # 3 false positive rate,index of ROC , 误报 (Type I error).
    # try:
    #     # res['FPR']='{}'.format(FP/(FP+TN))
    #     res['FPR']=FP/(FP+TN)
    # except ZeroDivisionError:
    #     res['FPR'] ='NA'
    # # 4 true positive rate,index of ROC
    # try:
    #     # res['TPR'] ='{}'.format(TP/(TP+FN))
    #     res['TPR'] =TP/(TP+FN)
    # except ZeroDivisionError:
    #     # writelog('have no P(normaly event)',file)
    #     res['TPR'] ='NA'
    # 5 accurate
    try:
        # res['acc'] = (TP+NN)/len(flags)
        res['acc'] = (TP+TN)/(total*64)
        # results['accurate'] = accurate
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['acc'] ='NA'
    #  recall same as TPR
    try:
        res['recall'] = TP/(TP+FN)
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['recall'] = 'NA'

    # F1
    try:
        res['F1'] = 2*TP/(2*TP+FP+FN)
    except ZeroDivisionError:
        # writelog('Error at get data,flags is None)',file)
        res['F1'] = 'NA'
    # false negative rate (Type II error).
    # try:
    #     # res['fnr']= '{}'.format(FN/(FN+TP))
    #     res['fnr']= FN/(FN+TP)
    # except ZeroDivisionError:
    #     # writelog('Error at get data,flags is None)',file)
    #     res['fnr'] = 'NA'

    t2 = time.time()
    text = ''
    for key, item in res.items():
        text += key + ':' + str(item) + ','
    writelog('len result|label:{}|{},source data label 1|label 0:{}|{}'.
             format(total*64,total*64,label1,label0),file)
    writelog(text,file)
    writelog('test case: {} had finshed module:{}'.format(file,logmark),file)
    writelog('time test spent :{}'.format(t2 - t1), file)
    writelog('*'*40,file)
    try:
        auc_score = roc_auc_score(np.array(y_true), np.array(y_pre_))
        # auc_score = np.squeeze(auc_score).item()
        return res, auc_score,fpr,tpr

    except:
        return res,fpr,tpr

def getModulesList(modules_path):
    """
    func: sort different modules saved at different epoch,sorted name list
    :param modules_path:
    :param mark:
    :return: different module file url(address) saved at different epoch,name list
    """
    modules = os.listdir(modules_path)
    pattern = re.compile(r'\d+\.?\d*')
    num_seq = []
    new_modul = []
    for module in modules:
        if '_D.pkl' in module:
            new_modul.append(module)
    modules = new_modul
    # print(modules)
    # print('len of modules: {}'.format(len(modules)))
    for i,module in enumerate(modules):
        jf = pattern.findall(module)
        if len(jf):
            num_seq.append(jf[-1])
        else:
            # 这里必须要取值大于epoch
            num_seq.append('100000')
            mark_D = i

    num_seq = list(map(int,num_seq))
    sort_seq = sorted(num_seq)
    modules_url = []

    for s in sort_seq:
        modules_url.append(os.path.join(modules_path,modules[num_seq.index(s)]))
    sort_seq = list(map(lambda x: str(x),sort_seq))
    # print('len of modules_url: {}'.format(len(modules_url)))
    # print('len of seqs: {}'.format(len(sort_seq)))
    return modules_url, sort_seq

def parallel_test_for_attackdataset(modules_dir,attack_name,dataset,flag=None):
    """
    func: test all kind of module trained by GAN or nets derived from GAN
    :param modules_dir: dir stored trained modules
    :param attack_name: the file name of a test dataset
    :param dataset: torch data loarder of a test dataset,hacking dataset,for test generative data it is a numpy array
    :return: None
    """
    # root dir of modules,have several types of GAN modules
    for name in os.listdir(modules_dir):
        module_url = os.path.join(module_path, name)
        result_url = os.path.join(result_path, name)
        # module_urls, seqs = getModulesList(module_url)

        print('\n------------------------{} test at {}--------------------------------------------------------------'.format(attack_name,name))
        if not os.path.exists(result_url):
            os.makedirs(result_url)

        os.chdir(result_url)
        # print(module_url)

        # get  all dir of trained modules whice belongs to a types of module and sorted as ascending sort
        module_urls,seqs = getModulesList(module_url)
        # print(seqs)
        ress = {}
        for i in columns_:
            ress[i] = []
        # draw auc_sorce curve,roc curve
        auc_socres = []
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        pic_url = './{}_picture'.format(name)
        if not os.path.exists(pic_url):
            os.makedirs(pic_url)
        result_dir = os.path.join(pic_url,'{}_roc_curve.png'.format(name))

        for i, url in list(zip(seqs,module_urls)):
            # add test generative data as attack data for testing
            # if flag != None:
            if flag.__class__==list:
                row = dataset.shape[0]//64
                Gen_data = np.empty((1,1,1,1))
                G_path = os.path.join(module_url,'{}_{}_G.pkl'.format(name,i))
                G_net = torch.load(G_path,map_location='cpu')
                # print('G_net:{}'.format(G_net.__class__))
                # print('row:{}'.format(row))
                # G_net = torch.load(G_path)

                # generative attack data
                for j in range(row):
                    # z_ = torch.rand((BATCH_SIZE, 62)).cuda()
                    z_ = torch.rand((BATCH_SIZE, Z_SIZE))
                    G_ = G_net(z_)
                    if j == 0:
                        # Gen_data = G_.data.cpu().numpy()
                        Gen_data = G_.data.numpy()
                        # print(Gen_data.shape)

                    else:
                        # Gen_data = np.concatenate((Gen_data,G_.data.cpu().numpy()),axis=0)
                        Gen_data = np.concatenate((Gen_data,G_.data.numpy()),axis=0)
                # print('dataset:{}'.format(dataset.shape))
                # print('Gen_data:{}'.format(Gen_data.shape))
                flag = np.array(flag).reshape((-1, 1))
                labels = np.concatenate((flag,np.ones((row*64,1))),axis=0)
                dataset = np.concatenate((np.expand_dims(dataset,axis=1),Gen_data),axis=0)
                # print('labels shape:{},dataset shape:{}'.format(labels.shape,dataset.shape))
                dataM = torch.from_numpy(dataset).float()
                labelM = torch.from_numpy(labels).float()
                TorchDataset = Data.TensorDataset(dataM, labelM)
                dataset = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

            objs = test_attfree(url,i,attack_name,dataset,name)#return dict test_attfree(path, logmark, file, test)
            # print(objs.__class__,'-'*10)
            try:
                (tes,auc_score,fpr, tpr) = objs
                # print('4'*50)
                auc_socres.append(auc_score)

            except:
                (tes,fpr, tpr) = objs#(path, logmark, file, test,name=None)
                # print('8'*50)
                # return
            # print('_'.join([str(i) for _ in range(20)]))
            if int(i) % 10 == 0 and len(fpr) != 0:
                imname = '{}'.format(i)
                plt.plot(fpr, tpr, label=imname)
            for key,item in tes.items():
                if key == 'NA':
                    key = None
                ress[key].append(item)
        # torch.empty_cache()
        # draw roc curve and auc_score curve
        plt.legend(loc='best')
        plt.savefig(result_dir)
        plt.show()
        plt.close()

        if len(auc_socres) != 0:
            result_dir = os.path.join(pic_url,'{}_auc_score_curve.png'.format(name))
            plt.figure(1)
            plt.xlabel('epoch')
            plt.ylabel('auc score')
            plt.title('auc score curve')
            plt.plot(seqs, auc_socres, label='auc score per epoch')
            plt.legend(loc='best')
            plt.savefig(result_dir)
            plt.show()
            plt.close()

        ress['Module No.'] = seqs
        summary_url = os.path.join(result_url,name+'_test_{}_analysis_summary.csv'.format(attack_name))
        data = pd.DataFrame(ress,columns=list(ress.keys()))
        data.to_csv(summary_url,sep=',',float_format='%.6f',header=True,index=True,mode='w',encoding='utf-8')


if __name__ == '__main__':

    # module_path = '/home/yyd/PycharmProjects/repeat_lab/repeat_lab'
    module_path = '/home/yyd/PycharmProjects/repeat_lab/repeat_gan'
    test_addr = '/home/yyd/dataset/hacking/one-hot-repeat-lab'
    result_path = '/home/yyd/PycharmProjects/repeat_lab/test_gan'
    # for i in os.listdir(module_path):
    #    models,seqs = getModulesList(os.path.join(module_path,i))
    #    print(seqs)
    #    print(i,len(models),len(seqs),models[0])
    # exit()
    """test Discriminator using attacking dataset and generative data"""
    print('start at:{}'.format(time.asctime(time.localtime(time.time()))))
    print('test dataset:%s'%test_addr)
    print('model sources:%s'%module_path)
    print('result stored:%s'%result_path,'\n')

    testData = []
    names = []
    dataloaders, names = read_dataset(
        root_path_=test_addr,target_type='pkl',read_target='all',usage='test',res_num=1,res_type='dataloader',bias_dataset='normal')
    if dataloaders.__class__ != list:
        parallel_test_for_attackdataset(module_path, 'mix', dataloaders)#(modules_dir,attack_name,dataset,flag=None):
    else:
        pool = multiprocessing.Pool(processes=len(names))
        for dat,name in list(zip(dataloaders,names)):
            parallel_test_for_attackdataset(module_path, name, dat)#(modules_dir,attack_name,dataset,flag=None):
        pool.close()
        pool.join()
    print('done at:{}'.format(time.asctime(time.localtime(time.time()))))
