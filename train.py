import torch,json
from nets import *
from utils import *
from tensorboardX import *
from read_func import *
from sklearn.metrics import roc_auc_score,roc_curve,auc
# INPUT_SIZE = 48

if __name__ == '__main__':

    addr = '/home/yyd/dataset/hacking/one-hot-repeat-lab'#attack data
    args = parse_args()
    select = []
    select.append(args.dataset)
    dataloader,names = read_dataset(root_path_=addr,target_type='pkl',read_target='all',usage='train',res_num=1,res_type='dataloader',bias_dataset='normal')#selected=['rpm'])
    valdata,_ = read_dataset(root_path_=addr,target_type='pkl',read_target='all',usage='validate',res_num=1,res_type='dataloader',bias_dataset='both')#selected=['rpm'])

    attack_type = '_'.join(names)
    train_type = 'CrossEnL_StepLR'
    # cnn = CNN(data_loader=dataloader,valdata=valdata,attack_type=attack_type,train_type=train_type)
    # cnn.train()
    gan = GAN(data_loader=dataloader,valdata=valdata,dataset_type=attack_type,train_type=train_type)
    gan.train()