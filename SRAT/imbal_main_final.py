import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10
from losses import *
from resnet import SupConResNet
import random
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('--reweight', default=100, type=float, help='reweight ratio. ')
parser.add_argument('--lam', default=2, type=float, help='weights between loss functions in adv training. ')
parser.add_argument('--gamma', default=1, type=float, help='weights for focal loss. ')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', help='model architecture: (default: resnet18)')
parser.add_argument('--imb_type', default="step", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')

#parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 2e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

def main():
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

    print('current seed num is:')
    print(args.seed)

    # create model
    model = SupConResNet()
    model = model.cuda()

    ## optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # cudnn.benchmark = True

    # Data loading code
    transform_adv_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_adv_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                     rand_number=args.seed, train=True, download=True,
                                     transform=transform_adv_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_adv_val)

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, pin_memory=True, shuffle=False)

    criterion_con = SupConLoss(temperature=0.07)

    ## trianing loop
    for epoch in range(args.start_epoch, args.epochs):
        adv_adjust_learning_rate(optimizer, epoch, args)

        if epoch < 160:
            per_cls_weights = [1.0] * 10
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        else:
            if args.reweight == 78.0:
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = [np.max(effective_num) / x for x in effective_num]
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                per_cls_weights = [np.max(cls_num_list) / float(x) for x in cls_num_list]
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

        ## assign classification losses
        if args.loss_type == 'CE':
            criterion_cla = nn.CrossEntropyLoss(weight=per_cls_weights).cuda()
        elif args.loss_type == 'LDAM':
            criterion_cla = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
        elif args.loss_type == 'Focal':
            criterion_cla = FocalLoss(weight=per_cls_weights, gamma=args.gamma).cuda()
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        adv_train(train_loader, model, criterion_cla, criterion_con, optimizer, epoch, args)

        ## evaluation
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            evaluation(test_loader, model)


def adv_train(train_loader, model, criterion_cla, criterion_con, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda()
        target = target.cuda()

        # compute output
        adv_input = pgd_attack(model, input, target, 8/255, 1., 0., 10, 2/255)
        f1, cla_output = model(adv_input)
        features = f1.unsqueeze(1)
        loss_con = criterion_con(features, target)
        loss_cla = criterion_cla(cla_output, target)
        loss = args.lam * loss_con + loss_cla

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, _ = accuracy(cla_output, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1,
                lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output, flush=True)


def evaluation(val_loader, model):
    top1_clean = AverageMeter('Acc@1', ':6.2f')
    top1_adv = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()
    clean_all_preds = []
    adv_all_preds = []
    all_targets = []
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute clean output
        _, clean_output = model(input)
        clean_acc1, _ = accuracy(clean_output, target, topk=(1, 1))
        top1_clean.update(clean_acc1[0], input.size(0))

        _, clean_pred = torch.max(clean_output, 1)
        clean_all_preds.extend(clean_pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        # compute adv output
        adv_input = pgd_attack(model, input, target, 8/255, 1., 0., 20, 2/255)
        _, adv_output = model(adv_input)
        adv_acc1, _ = accuracy(adv_output, target, topk=(1, 1))
        top1_adv.update(adv_acc1[0], input.size(0))

        _, adv_pred = torch.max(adv_output, 1)
        adv_all_preds.extend(adv_pred.cpu().numpy())

    # print clean output
    clean_cf = confusion_matrix(all_targets, clean_all_preds).astype(float)
    clean_cls_cnt = clean_cf.sum(axis=1)
    clean_cls_hit = np.diag(clean_cf)
    clean_cls_acc = clean_cls_hit / clean_cls_cnt
    clean_output = ('Test Clean Overall Results: Prec@1 {top1.avg:.3f}'.format(top1=top1_clean))
    clean_out_cls_acc = 'Test Clean Class Accuracy: %s' % (np.array2string(clean_cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))
    print(clean_output, flush=True)
    print(clean_out_cls_acc, flush=True)

    # print adv output
    adv_cf = confusion_matrix(all_targets, adv_all_preds).astype(float)
    adv_cls_cnt = adv_cf.sum(axis=1)
    adv_cls_hit = np.diag(adv_cf)
    adv_cls_acc = adv_cls_hit / adv_cls_cnt
    adv_result = ('Test Robust Overall Results: Prec@1 {top1.avg:.3f}'.format(top1=top1_adv))
    adv_out_cls_acc = 'Test Robust Class Accuracy: %s' % (np.array2string(adv_cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))
    print(adv_result, flush=True)
    print(adv_out_cls_acc, flush=True)


def pgd_attack(model, X, y, epsilon, clip_max, clip_min, num_steps, step_size):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        _, pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd


def adv_adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()