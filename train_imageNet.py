import os
import sys
import numpy as np
from time import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from model import NetworkImageNet as Network

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--path', type=str)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
args = parser.parse_args()

PATH = args.path
model_path = os.path.join(PATH, 'checkpoint')
args.data_dir = os.path.join(PATH, 'data')
os.makedirs(model_path, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)
CLASSES = 1000

writer = SummaryWriter(os.path.join(PATH, 'runs/darts_search'))

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print(f'gpu device = {args.gpu}')

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    if args.parallel:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    try:
        checkpoint = torch.load(os.path.join(model_path, 'weights_search.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_old = checkpoint['epoch']
        print('Load previous model at : ', model_path)
    except:
        epoch_old = -1
        print('Training new model!')
    print(f"param size = {utils.count_parameters_in_MB(model)}MB")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'val')
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_data = dset.ImageFolder(
        valid_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    best_acc_top1 = 0
    for epoch in range(epoch_old, args.epochs):
        print(f'epoch {epoch} lr {scheduler.get_last_lr()[0]}')
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        start_train = time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        end_train = time()
        print(f'train_acc: {train_acc.item()}%, time:{end_train - start_train}s')
        writer.add_scalar('train accuracy', train_acc.item(), epoch)
        writer.add_scalar('train loss', train_obj.item(), epoch)
        writer.add_scalar('train time', end_train - start_train, epoch)

        start_valid = time()
        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        end_valid = time()
        print(f'valid_acc_top1: {valid_acc_top1.item()}%, valid_acc_top5: {valid_acc_top5.item()}%, time:{end_valid - start_valid}s')
        writer.add_scalar('valid accuracy', valid_acc_top1.item(), epoch)
        writer.add_scalar('valid loss', valid_obj.item(), epoch)
        writer.add_scalar('valid time', end_valid - start_valid, epoch)

        scheduler.step()
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            torch.save({
              'epoch': epoch,
              'best_valid_acc': best_acc_top1,
              'model_state_dict': model.state_dict(),
            }, os.path.join(model_path, 'best_model_imageNet.pt'))

        utils.save(model, epoch, optimizer, scheduler, os.path.join(model_path, 'weights_imageNet.pt'))

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda()
        input = input.cuda()
        input = Variable(input)
        target = Variable(target)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            print(f'train {step}: {objs.avg.item}   {top1.avg.item}%  {top5.avg.item}%')

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                print(f'valid {step}: {objs.avg.item}   {top1.avg.item}%  {top5.avg.item}%')

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
