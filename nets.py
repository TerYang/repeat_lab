# -*- coding: utf-8 -*-
# @Time   : 19-4-25 上午10:59
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse,json,utils
from utils import *
import matplotlib.pyplot as plt
from tensorboardX import *
from sklearn.metrics import roc_curve,roc_auc_score
# from dataloader import dataloader
# from readDataToGAN import *

def parse_args():
    """parsing and configuration"""
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    # parser.add_argument('--gan_type', type=str, default='None',#'ACGAN',#'BEGAN',#'GAN',#'LSGAN',#default='GAN',
    #                     choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
    #                     help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='normal', choices=['dos', 'rpm', 'fuzzy', 'gear', 'normal'],
                        help='The name of dataset')
    # parser.add_argument('--dataset', type=str, default='dos', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
    #                     help='The name of dataset')

    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=250, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=48, help='The size of input image')
    # parser.add_argument('--save_dir', type=str, default='models',
    #                     help='Directory name to save the model')
    parser.add_argument('--save_dir', type=str, default='repeat_gan', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.05)
    parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=False)

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    # --save_dir
    if not os.path.exists(args.save_dir):
        # os.makedirs(os.path.join(os.getcwd(),args.save_dir))
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

class generator(nn.Module):
    """
       convtranspose2d
        #   1.  s =1
        #     o' = i' - 2p + k - 1
        #   2.  s >1
        # o = (i-1)*s +k-2p+n
        # n =  output_padding,p=padding,i=input dims,s=stride,k=kernel
    """
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Tanh(),
        )
        initialize_weights(self)

        self.fc = nn.Sequential(
            nn.Linear(256, 3*4*512),
            nn.ReLU(),
        )
    def forward(self, input):
        x = self.fc(input)
        # x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = x.view(-1, 512, 4, 3)
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # input 64*48
    def __init__(self, input_dim=1, output_dim=1, input_size=64):
        super(discriminator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            # nn.Conv2d(self.input_dim, 8, (2,1), (2,1)),#torch.Size([64, 8, 32, 48])
            nn.Conv2d(self.input_dim, 64, (4,5), (2,1),(1,2)),#torch.Size([64, 8, 32, 48])
            nn.ReLU(),
            # nn.Conv2d(8, 16, (2, 1), (2, 1)),
            nn.Conv2d(64, 128, (4, 5), (2, 1), (1, 2)),  # torch.Size([64, 8, 32, 48])

            # nn.Conv2d(8, 16, (4,1), (2,1),(1,0)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16*(16 * 3), 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 16*(16 * 3))
        x = self.fc(x)
        return x


class CNN(object):
    def __init__(self, data_loader, valdata, attack_type, train_type):
        args = parse_args()

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.gpu_mode = args.gpu_mode
        self.input_size = args.input_size
        # self.lambda_ = 0.25
        self.train_hist = {}
        self.dataset = attack_type
        if len(train_type):
            self.model_name = self.__class__.__name__ + '_' + train_type
        else:
            self.model_name = self.__class__.__name__

        self.data_loader = data_loader
        self.valdata = valdata

        data = next(iter(self.data_loader))[0]

        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # Step LR
        # self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, 20, gamma=0.1, last_epoch=-1)
        self.D_scheduler = optim.lr_scheduler.StepLR(self.D_optimizer, 20, gamma=0.1, last_epoch=-1)

        if self.gpu_mode:
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            # self.CEL = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            # self.CEL = nn.CrossEntropyLoss()
        self.writer = SummaryWriter()  # log_dir=log_dir,
        self.X = 0
        print('Training {},started at {}'.format(self.model_name, time.asctime(time.localtime(time.time()))), end=',')

    def train(self):
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['per_epoch_loss'] = []
        self.D.train()
        print('training start!!,data set:{},epoch:{}'.format(self.dataset, self.epoch))
        # return
        start_time = time.time()

        auc_scores = []
        epochs = []

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        imname = '_'.join((self.model_name, 'roc_curve'))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')

        # stored_url = '/home/yyd/PycharmProjects/repeat_lab/repeat_lab/DoS/CNN_CrossEnL_StepLR'
        # url = os.path.join(stored_url,self.model_name+'_245_D.pkl')

        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            # if epoch == 246:
            #     self.D = torch.load(url)
            self.D.train()
            for iter, (x_, label_) in enumerate(self.data_loader):
                # x_ = x_[0]
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                # if iter == 0:
                #     input_data = Variable(x_)

                if self.gpu_mode:
                    x_, label_ = x_.cuda(), label_.cuda()
                self.D_optimizer.zero_grad()

                # print(x_.data.cpu().size())

                D_real = self.D(x_)

                D_loss = self.BCE_loss(D_real, label_)
                # D_loss = self.CEL(D_real,label_)

                # get loss of the end iter of train in every epoch
                if iter == self.data_loader.dataset.__len__() // self.batch_size - 1:
                    self.train_hist['per_epoch_loss'].append(D_loss.item())

                self.train_hist['D_loss'].append(D_loss.item())
                D_loss.backward()
                self.D_optimizer.step()
                if ((iter + 1) % 200) == 0:
                    self.writelog("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, D_lr: %.8f" %
                                  ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                                   D_loss.item(), self.D_optimizer.param_groups[0]['lr']))
                    self.writer.add_scalar('D_loss', D_loss.item(), self.X)
                    self.X += 1

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            if epoch % 5 == 0:
                self.load_interval(epoch)

            res, y_pre_, all_l_ = validate(self.D,self.valdata)  # def validate(model,data_loader=None,data=None,label=None)
            # print(len(all_l_),len(y_pre_),all_l_.__class__,y_pre_.__class__)
            fpr, tpr, _ = roc_curve(np.array(all_l_), np.array(y_pre_))
            auc_score = roc_auc_score(np.array(all_l_), np.array(y_pre_))
            auc_score = np.squeeze(auc_score).item()
            # print('auc_score:',auc_score,auc_score.__class__)
            print('auc_score:', auc_score)

            self.D_scheduler.step(epoch)
            self.D.cuda()

            if epoch % 30 == 0:
                # self.writer.add_scalar('roc_curve', tpr,fpr)
                plt.plot(fpr, tpr, label=str(epoch))

                self.writer.add_scalar('auc_score', auc_score, epoch)
                auc_scores.append(auc_score)
                epochs.append(epoch)

        # with self.writer:
        #     self.writer.add_graph(self.D)

        result_dir = os.path.join(os.getcwd(), self.save_dir, self.dataset, imname + '.png')
        plt.legend(loc='best')
        plt.savefig(result_dir)
        plt.show()
        plt.close()

        # draw and store auc score image
        imname = '_'.join((self.model_name, 'auc_score'))
        plt.figure(1)
        plt.plot(auc_scores, epochs, label=imname)
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.title('auc_score')
        plt.legend(loc='best')
        result_dir = os.path.join(os.getcwd(), self.save_dir, self.dataset, imname + '.png')
        plt.savefig(result_dir)
        plt.show()
        plt.close()

        # close all
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (
        np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))

        # self.writelog("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),self.epoch, self.train_hist['total_time'][0]))

        print("Training finish!... save training results")

        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        with open(os.path.join(save_dir, self.model_name + '_train_hist.json'), "a") as f:
            json.dump(self.train_hist, f)

        self.writer.export_scalars_to_json(os.path.join(save_dir, self.model_name + '.json'))
        self.writer.close()

        self.load_interval(self.epoch)
        # draw loss picture
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def load_interval(self, epoch):
        save_dir = os.path.join(os.getcwd(), self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        # torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

    def writelog(self, content):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_log = os.path.join(save_dir, 'train_records.txt')

        with open(save_log, 'a', encoding='utf-8') as f:
            f.writelines('\n' + content + '\n')
            print(content, end=',')

class GAN(object):
    # def __init__(self, args):#,data_loader,valdata):
    def __init__(self, data_loader,valdata,dataset_type,train_type):
        # parameters
        args = parse_args()
        self.dataset = dataset_type
        self.data_loader = data_loader
        self.valdata = valdata
        self.model_name = self.__class__.__name__ + '_' + train_type
        # parameters

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.input_size = args.input_size
        self.z_dim = 256
        self.train_hist = {}

        """dataset"""
        data = next(iter(self.data_loader))[0]#torch.Size([64, 1, 64, 21])

        # networks ini
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # lr_scheduler
        # self.G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer, mode='max', factor=0.1, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)
        # self.D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer, mode='max', factor=0.1, patience=4, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

        # same epoch interval reduce lr
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, 20, gamma=0.1, last_epoch=-1)
        self.D_scheduler = optim.lr_scheduler.StepLR(self.D_optimizer, 20, gamma=0.1, last_epoch=-1)

        # ExponentialLR
        # self.G_scheduler = optim.lr_scheduler.ExponentialLR(self.G_optimizer,0.9)
        # self.D_scheduler = optim.lr_scheduler.ExponentialLR(self.D_optimizer,0.9)

        self.y_real_, self.y_fake_ = torch.zeros(self.batch_size, 1), torch.ones(self.batch_size, 1)

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.randn((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

        self.writer = SummaryWriter()#log_dir=log_dir,
        self.X = 0
        print('Training {},started at {}'.format(self.model_name, time.asctime(time.localtime(time.time()))),end=',')

    def train(self):

        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['per_epoch_D_loss'] = []
        self.train_hist['per_epoch_G_loss'] = []
        print('{} training start!!,data set:{},epoch:{}'.format(self.model_name, self.dataset,self.epoch))

        start_time = time.time()

        auc_scores = []
        epochs = []

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        imname = '_'.join((self.model_name, 'roc_curve'))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        # model = '/home/yyd/PycharmProjects/repeat_lab/repeat_gan/DoS_gear_RPM_Fuzzy_GAN_CrossEnL_StepLR'

        for epoch in range(self.epoch):

            # if epoch == 66:
            #     # self.G = torch.load(os.path.join(model,'GAN_CrossEnL_StepLR_65_G.pkl'))
            #     self.D = torch.load(os.path.join(model,'GAN_CrossEnL_StepLR_65_D.pkl'))
            self.G.train()
            self.D.train()
            epoch_start_time = time.time()
            # for iter, (x_, _) in enumerate(self.data_loader):
            for iter, (x_,l_) in enumerate(self.data_loader):
                # x_ = x_[0]
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.randn((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ ,l_ = x_.cuda(), z_.cuda(),l_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                # D_real_loss = self.BCE_loss(D_real, self.y_real_)
                D_real_loss = self.BCE_loss(D_real, l_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                # G_loss = self.BCE_loss(D_fake, self.y_real_)
                G_loss = self.BCE_loss(D_fake, l_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if iter == self.data_loader.dataset.__len__() // self.batch_size - 1:
                    self.train_hist['per_epoch_D_loss'].append(D_loss.item())
                    self.train_hist['per_epoch_G_loss'].append(G_loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f,lr_G:%.10f,lr_D:%.10f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item()
                          ,G_loss.item(),self.G_optimizer.param_groups[0]['lr'],self.D_optimizer.param_groups[0]['lr']))
                    self.writer.add_scalar('G_loss', G_loss.item(), self.X)
                    # writer.add_scalar('G_loss', -G_loss_D, X)
                    self.writer.add_scalar('D_loss', D_loss.item(), self.X)
                    self.writer.add_scalars('cross loss', {'G_loss': D_loss.item(),
                                                      'D_loss': D_loss.item()}, self.X)
                    self.X += 1

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            if epoch % 5 == 0:
                self.load_interval(epoch)

                res, y_pre_, all_l_ = validate(self.D, self.valdata)  # def validate(model,data_loader=None,data=None,label=None)
                # print(len(all_l_),len(y_pre_),all_l_.__class__,y_pre_.__class__)
                fpr, tpr, _ = roc_curve(np.array(all_l_), np.array(y_pre_))

                auc_score = roc_auc_score(np.array(all_l_), np.array(y_pre_))
                auc_score = np.squeeze(auc_score).item()
                # print('auc_score:',auc_score,auc_score.__class__)
                print('auc_score:', auc_score,end=',')
                self.D.cuda()
                acc_G = self.validate_G(self.valdata.dataset.tensors[0].shape[0] // 2)

                if epoch % 10 == 0:
                    # draws roc curve per 30 epoch and records auc_score
                    plt.plot(fpr, tpr, label=str(epoch))
                    self.writer.add_scalar('auc_score', auc_score, epoch)
                    auc_scores.append(auc_score)
                    epochs.append(epoch)

            # update learning-rate
            self.D_scheduler.step(epoch)
            self.G_scheduler.step(epoch)

        result_dir = os.path.join(os.getcwd(), self.save_dir, self.dataset, imname + '.png')
        plt.legend(loc='best')
        plt.savefig(result_dir)
        plt.show()
        plt.close()

        # draw and store auc score image
        imname = '_'.join((self.model_name, 'auc_score'))

        plt.figure(1)
        plt.plot(auc_scores, epochs, label=imname)
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.title('auc_score')
        plt.legend(loc='best')

        result_dir = os.path.join(os.getcwd(), self.save_dir, self.dataset, imname + '.png')
        plt.savefig(result_dir)
        plt.show()
        plt.close()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),self.epoch, self.train_hist['total_time'][0]))
        # self.writelog("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),self.epoch, self.train_hist['total_time'][0]))

        print("Training finish!... save training results")

        save_dir = os.path.join(self.save_dir, '_'.join([self.dataset,self.model_name]))

        with open(os.path.join(save_dir, self.model_name + '_train_hist.json'), "a") as f:
            json.dump(self.train_hist, f)

        self.writer.export_scalars_to_json(os.path.join(save_dir, self.model_name + '.json'))
        self.writer.close()

        self.load_interval(self.epoch)

        # loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        loss_plot(self.train_hist, save_dir, self.model_name)

    def load_interval(self, epoch):
        save_dir = os.path.join(os.getcwd(), self.save_dir, '_'.join([self.dataset,self.model_name]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        # torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))
        if epoch%10 == 0:
            torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))

    def writelog(self, content):
        save_dir = os.path.join(self.save_dir,'_'.join([self.dataset,self.model_name]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_log = os.path.join(save_dir, 'train_records.txt')

        with open(save_log, 'a', encoding='utf-8') as f:
            f.writelines('\n' + content + '\n')
            print(content, end=',')

    def validate_G(self, size):
        # validate G
        self.G.eval()
        acc_G = 0
        sum_all = 0
        for i in range(size // 64):
            z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                z_ = z_.cuda()
            G_ = self.G(z_)
            D_fake = self.D(G_)
            # print(D_fake.__class__)
            D_fake = np.squeeze(D_fake.data.cpu().numpy(), axis=1)
            # D_fake = D_fake.tolist()
            f = lambda x: 1 if x > 0.5 else 0
            ll = list(map(f, D_fake.tolist()))
            acc_G += ll.count(1)
            sum_all += len(ll)
        zeros = sum_all - acc_G
        ones = acc_G
        print('--G: size:%d,zeros:%d,ones:%d' % (sum_all, zeros, ones), end=',')
        print('acc:%.6f,judged as 1.' % (ones / sum_all))
        return ones / sum_all


# t = torch.ones((64,1,64,48))
# print(t.size())
# g = discriminator(1,1,64)
# l = g(t)
# print(l.size())

# qq = torch.randn((64,256))
# g = generator()
# l = g(qq)
# print(l.size())
# qq = torch.randn((1,10))
# print(qq)