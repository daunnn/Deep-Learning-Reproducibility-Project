import shutil
import warnings
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
warnings.filterwarnings("ignore")
import torch.utils.data as data
import os
import argparse
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from models.PosterV2_3cls import *

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
# 수정
parser.add_argument('--data', type=str, default=r'/deep_project/modify_face_dataset')
# parser.add_argument('--data_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'CAER-S'],
#                         type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
# epoch 수정: 100 -> 50
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# batchsize 수정: 64 -> 128
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

from torchvision import datasets, transforms
from torchvision.transforms import functional as F

# 수정(추가)

def plot_confusion_matrix(cm, class_names):
    """
    cm: confusion matrix 값을 넘겨 받을 매개변수
    class_names: confusion matrix의 레이블 이름 리스트
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    matrix_name = time_str + 'confusion_matrix'
    plt.savefig(f'./log/{matrix_name}.png', format='png')
    plt.show()
            
class MaintainAspectRatioResize:
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, img):
        aspect_ratio = img.width / img.height
        if img.width < img.height:
            w = self.desired_size
            h = int(self.desired_size / aspect_ratio)
        else:
            h = self.desired_size
            w = int(self.desired_size * aspect_ratio)
        return F.resize(img, (h, w))

class CustomPad:
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, img):
        current_size = img.size
        padding = [0, 0, 0, 0] # left, top, right, bottom

        # width padding
        if current_size[0] < self.desired_size[0]:
            diff = self.desired_size[0] - current_size[0]
            padding[0] = diff // 2
            padding[2] = diff - diff // 2

        # height padding
        if current_size[1] < self.desired_size[1]:
            diff = self.desired_size[1] - current_size[1]
            padding[1] = diff // 2
            padding[3] = diff - diff // 2

        return F.pad(img, padding)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    # 수정
    model = pyramid_trans_expr2(img_size=224, num_classes=3)

    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    
    # valid --> test 로 수정
    testdir = os.path.join(args.data, 'test')
    
    valdir = os.path.join(args.data, 'valid')

    
    def is_valid_file(path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return path.lower().endswith(valid_extensions)
    
    
    
    # 수정
    # Resize((224, 224)) -> transforms.Resize(256), transforms.CenterCrop(224),
#     train_dataset = datasets.ImageFolder(traindir,
#                                          transforms.Compose([transforms.Resize((224, 224),
#                                                              transforms.RandomHorizontalFlip(),
#                                                              transforms.ToTensor(),
#                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                   std=[0.229, 0.224, 0.225]),
#                                                              transforms.RandomErasing(scale=(0.02, 0.1))]))
    # 수정
    # 변환 시퀀스 설정
    transform2 = transforms.Compose([
        MaintainAspectRatioResize(224),
        CustomPad((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])
    transform1 = transforms.Compose([
        MaintainAspectRatioResize(224),
        CustomPad((224, 224)),
        transforms.RandomHorizontalFlip(),
        # Add additional augmentation methods here
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])

    # ImageFolder를 사용하여 데이터셋 불러오기
    train_dataset = datasets.ImageFolder(traindir, transform=transform1, is_valid_file=is_valid_file)

    test_dataset = datasets.ImageFolder(testdir, transform = transform2, is_valid_file=is_valid_file)
    
    val_dataset = datasets.ImageFolder(valdir, transform=transform2, is_valid_file=is_valid_file)
    
#     train_dataset = datasets.ImageFolder(traindir, is_valid_file=is_valid_file)

#     test_dataset = datasets.ImageFolder(testdir, is_valid_file=is_valid_file)
    
#     val_dataset = datasets.ImageFolder(valdir, is_valid_file=is_valid_file)
    
    
    # loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    
    
    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to()
            print(f'best_acc:{best_acc}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        validate(val_loader, model, criterion, args)
        print("I'm here!")
        return

    matrix = None

    for epoch in range(args.start_epoch, args.epochs):
#         #수정 ealy stopping 추가
#         early_stopping = EarlyStopping(patience=10, verbose=True)

        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los, output, target, D = validate(val_loader, model, criterion, args)

        scheduler.step()
        

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder1.update(output, target)

        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())

        if is_best:
            matrix = D

        print('Current best matrix: ', matrix)

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)
#         early_stopping(val_los, model)
    
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    
        
    # Confusion matrix calculations and plotting
    if matrix is not None:
        recorder1.plot_confusion_matrix(matrix, title='Confusion Matrix')
    
    



def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        # print(images.shape)
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer.first_step(zero_grad=True)
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.second_step(zero_grad=True)

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    # 수정
    model.eval()
    D = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            loss = criterion(output, target)
            
#             # 디버깅 중
#             print(f"Target labels: {target}")

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            topk = (1,) #2
            # """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
            #수정 maxk = 1
#                 maxk = 1
                # batch_size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

            output = pred
            target = target.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()

            im_re_label = np.array(target)
            im_pre_label = np.array(output)
            y_true = im_re_label.flatten() #y_ture 수정
            im_re_label.transpose()
            y_pred = im_pre_label.flatten()
            im_pre_label.transpose()

            # 수정
            C = metrics.confusion_matrix(y_true, y_pred, labels=[0,1,2]) #y_ture 수정
            D += C
            

            if i % args.print_freq == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    print(D)
    plot_confusion_matrix(np.array(D), class_names=['E01', 'E02', 'E03'])

    return top1.avg, losses.avg, output, target, D

def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        best_state = state.pop('optimizer')
        torch.save(best_state, args.best_checkpoint_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 수정
labels = [0,1,2]


class RecorderMeter1(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target
        

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        y_true = self.y_true
        y_pred = self.y_pred

        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 주석 처리
#         plt.figure(figsize=(12, 8), dpi=120)

#         ind_array = np.arange(len(labels))
#         x, y = np.meshgrid(ind_array, ind_array)
#         for x_val, y_val in zip(x.flatten(), y.flatten()):
#             c = cm_normalized[y_val][x_val]
#             if c > 0.01:
#                 plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
#         # offset the tick
#         tick_marks = np.arange(len(labels))
#         plt.gca().set_xticks(tick_marks, minor=True)
#         plt.gca().set_yticks(tick_marks, minor=True)
#         plt.gca().xaxis.set_ticks_position('none')
#         plt.gca().yaxis.set_ticks_position('none')
#         plt.grid(True, which='minor', linestyle='-')
#         plt.gcf().subplots_adjust(bottom=0.15)
        
        
        #수정
        # 이 부분에서 'ConfusionMatrixDisplay'를 사용하여 혼동 행렬을 표시합니다.
        cm_display = ConfusionMatrixDisplay(cm_normalized, display_labels=labels)
        cm_display.plot(cmap=cmap, values_format=".2f")

#         plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
    #수정
        matrix_name = time_str + 'confusion_matrix'
        plt.savefig(f'./log/{matrix_name}.png', format='png')
        # fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('Saved figure')
        plt.show()

    def matrix(self):
        target = self.y_true
        output = self.y_pred
        im_re_label = np.array(target)
        im_pre_label = np.array(output)
        y_true = im_re_label.flatten() #y_ture 수정
        # im_re_label.transpose()
        y_pred = im_pre_label.flatten()
        im_pre_label.transpose()

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy curve of train/valid'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

#         y_axis[:] = self.epoch_losses[:, 0]
#         plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)

#         y_axis[:] = self.epoch_losses[:, 1]
#         plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
