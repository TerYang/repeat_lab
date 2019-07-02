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

GPU_ENBLE = False
DATA_TYPE = 'attack_free'#'hacking_datasets'
TEST_Gen = False # test Generator generates data at the same time

columns_ = ['pre','N_pre','F1','acc','recall',]
BATCH_SIZE = 64

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
    Dnet = torch.load(path,map_location='cpu')
    if GPU_ENBLE:
        pass
    else:
        Dnet = Dnet.cpu()

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
    pic_url = './{}_picture'.format(file)
    if not os.path.exists(detail_url):
        os.makedirs(detail_url)
        os.makedirs(pic_url)
    url_numpy = os.path.join(detail_url,'{}_test_at_{}.csv'.format(logmark,file))
    y_test = []
    y_pre_ = []

    label0 = 0
    label1 = 0
    for iter, (x_, label_) in enumerate(test):
        if iter == test.dataset.__len__() // 64:
            total = iter
            break
        if GPU_ENBLE:
            x_ = x_.cuda()
            try:
                Results = Dnet(x_)
                result = Results.data.cpu().numpy()
            except:
                try:
                    Results, _ = Dnet(x_)
                    result = Results.data.cpu().numpy()
                except:
                    print('path:', path,'file:',file)
        else:
            try:
                Results = Dnet(x_)
                result = Results.data.numpy()
                # print(len(result))
            except:
                # try:
                Results, _ = Dnet(x_)
                result = Results.data.numpy()
                # except:
                #     print('path:', path,'file:',file)
        result = np.squeeze(result).tolist()
        label = np.squeeze(label_.data.numpy()).tolist()
        y_pre_.extend(result)
        y_test.extend(label)
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
    fpr,tpr,_ = roc_curve(y_test,y_pre_)
    # auc_score = roc_auc_score(np.array(y_test), np.array(y_pre_))
    # auc_score = np.squeeze(auc_score).item()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    imname = '{}_test_at_{}_roc_curve'.format(logmark,file)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')

    result_dir = os.path.join(pic_url,'{}_test_at_{}_roc_curve.png'.format(logmark,file))
    plt.legend(loc='best')
    plt.savefig(result_dir)
    plt.show()
    plt.close()

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
    return res

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
            num_seq.append(jf[0])
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
        module_urls, seqs = getModulesList(module_url)

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
                for i in range(row):
                    # z_ = torch.rand((BATCH_SIZE, 62)).cuda()
                    z_ = torch.rand((BATCH_SIZE, 62))
                    G_ = G_net(z_)
                    if i == 0:
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

            # tes = test_attfree(url,i,attack_name,dataset,name)#return dict test_attfree(path, logmark, file, test)
            tes = test_attfree(url,i,attack_name,dataset,name)#(path, logmark, file, test,name=None)
            for key,item in tes.items():
                if key == 'NA':
                    key = None
                ress[key].append(item)
        ress['Module No.'] = seqs
        summary_url = os.path.join(result_url,name+'_test_{}_analysis_summary.csv'.format(attack_name))
        data = pd.DataFrame(ress,columns=list(ress.keys()))
        data.to_csv(summary_url,sep=',',float_format='%.6f',header=True,index=True,mode='w',encoding='utf-8')


if __name__ == '__main__':

    module_path = '/home/yyd/PycharmProjects/repeat_lab/repeat_lab'
    test_addr = '/home/yyd/dataset/hacking/one-hot-repeat-lab'
    result_path = '/home/yyd/PycharmProjects/repeat_lab/test_and_CMP'

    """test Discriminator using attacking dataset and generative data"""
    print('start at:{}'.format(time.asctime(time.localtime(time.time()))))
    print('test dataset:%s'%test_addr)
    print('model sources:%s'%module_path)
    print('result stored:%s'%result_path,'\n')

    testData = []
    names = []
    dataloaders, names = read_dataset(
        root_path_=test_addr,target_type='pkl',read_target='all',usage='test',res_num=4,res_type='dataloader',bias_dataset='attack')

    pool = multiprocessing.Pool(processes=len(names))
    for dat,name in list(zip(dataloaders,names)):
        parallel_test_for_attackdataset(module_path, name, dat)#(modules_dir,attack_name,dataset,flag=None):
    pool.close()
    pool.join()
    print('done at:{}'.format(time.asctime(time.localtime(time.time()))))
