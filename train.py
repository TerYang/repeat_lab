import torch,json
from nets import *
from utils import *
from tensorboardX import *
from read_func import *
from sklearn.metrics import roc_auc_score,roc_curve,auc
# INPUT_SIZE = 48

def parse_args():
    """parsing and configuration"""
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    # parser.add_argument('--gan_type', type=str, default='None',#'ACGAN',#'BEGAN',#'GAN',#'LSGAN',#default='GAN',
    #                     choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
    #                     help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='dos', choices=['dos', 'rpm', 'fuzzy', 'gear', 'normal'],
                        help='The name of dataset')
    # parser.add_argument('--dataset', type=str, default='dos', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
    #                     help='The name of dataset')

    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=250, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=48, help='The size of input image')
    # parser.add_argument('--save_dir', type=str, default='models',
    #                     help='Directory name to save the model')
    parser.add_argument('--save_dir', type=str, default='repeat_lab', help='Directory name to save the model')
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

class CNN(object):
    def __init__(self,data_loader,valdata,attack_type,train_type):
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
        self.writer = SummaryWriter()#log_dir=log_dir,
        self.X = 0
        print('Training {},started at {}'.format(self.model_name, time.asctime(time.localtime(time.time()))),end=',')

    def train(self):
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['per_epoch_loss'] = []
        self.D.train()
        print('training start!!,data set:{},epoch:{}'.format(self.dataset,self.epoch))
        # return
        start_time = time.time()

        auc_scores = []
        epochs = []

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        imname = '_'.join((self.model_name,'roc_curve'))
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
            for iter, (x_,label_) in enumerate(self.data_loader):
                # x_ = x_[0]
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                # if iter == 0:
                #     input_data = Variable(x_)

                if self.gpu_mode:
                    x_,label_ = x_.cuda(),label_.cuda()
                self.D_optimizer.zero_grad()

                # print(x_.data.cpu().size())

                D_real = self.D(x_)

                D_loss = self.BCE_loss(D_real,label_)
                # D_loss = self.CEL(D_real,label_)

                # get loss of the end iter of train in every epoch
                if iter == self.data_loader.dataset.__len__()//self.batch_size -1:
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

            res,y_pre_,all_l_ = validate(self.D, self.valdata)#def validate(model,data_loader=None,data=None,label=None)
            # print(len(all_l_),len(y_pre_),all_l_.__class__,y_pre_.__class__)
            fpr, tpr, _ = roc_curve(np.array(all_l_), np.array(y_pre_))
            auc_score = roc_auc_score(np.array(all_l_), np.array(y_pre_))
            auc_score = np.squeeze(auc_score).item()
            # print('auc_score:',auc_score,auc_score.__class__)
            print('auc_score:',auc_score)

            self.D_scheduler.step(epoch)
            self.D.cuda()

            if epoch % 30 == 0:
                # self.writer.add_scalar('roc_curve', tpr,fpr)
                plt.plot(fpr,tpr,label=str(epoch))

                self.writer.add_scalar('auc_score', auc_score, epoch)
                auc_scores.append(auc_score)
                epochs.append(epoch)

        # with self.writer:
        #     self.writer.add_graph(self.D)

        result_dir = os.path.join(os.getcwd(),self.save_dir, self.dataset, imname + '.png')
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
        result_dir = os.path.join(os.getcwd(),self.save_dir, self.dataset, imname + '.png')
        plt.savefig(result_dir)
        plt.show()
        plt.close()

        # close all
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))

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

    def load_interval(self,epoch):
        save_dir = os.path.join(os.getcwd(),self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存模型
        # torch.save(self.G, os.path.join(save_dir, self.model_name + '_{}_G.pkl'.format(epoch)))#dictionary ['bias', 'weight']
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_{}_D.pkl'.format(epoch)))

    def writelog(self, content):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_log = os.path.join(save_dir,'train_records.txt')

        with open(save_log,'a',encoding='utf-8') as f:
            f.writelines('\n'+content + '\n')
            print(content,end=',')

    # def validate_G(self, size):
    #     # validate G
    #     self.G.eval()
    #     acc_G = 0
    #     sum_all = 0
    #     for i in range(size // 64):
    #         z_ = torch.rand((self.batch_size, self.z_dim))
    #         if self.gpu_mode:
    #             z_ = z_.cuda()
    #         G_ = self.G(z_)
    #         D_fake = self.D(G_)
    #         # print(D_fake.__class__)
    #         D_fake = np.squeeze(D_fake.data.cpu().numpy(), axis=1)
    #         # D_fake = D_fake.tolist()
    #         f = lambda x: 1 if x > 0.5 else 0
    #         ll = list(map(f, D_fake.tolist()))
    #         acc_G += ll.count(1)
    #         sum_all += len(ll)
    #     zeros = sum_all - acc_G
    #     ones = acc_G
    #     print('--G: size:%d,zeros:%d,ones:%d' % (sum_all, zeros, ones), end=',')
    #     print('acc:%.6f,judged as 1.' % (ones / sum_all))
    #     return ones / sum_all
if __name__ == '__main__':

    addr = '/home/yyd/dataset/hacking/one-hot-repeat-lab'#attack data
    args = parse_args()
    select = []
    select.append(args.dataset)
    dataloader,names = read_dataset(root_path_=addr,target_type='pkl',read_target='select',usage='train',res_num=1,res_type='dataloader',bias_dataset='both',selected=select)#selected=['rpm'])
    valdata,names = read_dataset(root_path_=addr,target_type='pkl',read_target='select',usage='validate',res_num=1,res_type='dataloader',bias_dataset='both',selected=select)#selected=['rpm'])

    attack_type = '_'.join(names)
    train_type = 'CrossEnL_StepLR'
    cnn = CNN(data_loader=dataloader,valdata=valdata,attack_type=attack_type,train_type=train_type)
    cnn.train()