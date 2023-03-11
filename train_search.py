import os
import sys
import time
import argparse
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from time import time
from torch.utils.tensorboard import SummaryWriter
from model_search import Network
from utils import *
from utils import _data_transforms_cifar10
from architect import *


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--path', type=str)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

PATH = args.path
model_path = os.path.join(PATH, 'checkpoint')
args.data_dir = os.path.join(PATH, 'data')
os.makedirs(model_path, exist_ok=True)
os.makedirs(args.data_dir, exist_ok=True)
CIFAR_CLASSES = 10

writer = SummaryWriter(os.path.join(PATH, 'runs/darts_search'))

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
    print('gpu device = ', args.gpu)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )
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
    print(f"param size = {count_parameters_in_MB(model)}MB")

    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    architect = Architect(model, args)

    best_valid_acc = 0
    for epoch in range(epoch_old+1, args.epochs):
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning rate', lr, epoch)
        print('epoch: ', str(epoch), ', lr: ', str(lr))

        genotype = model.genotype()
        print('genotype = ', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        start_train = time()
        train_acc1, train_acc5, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        end_train = time()

        print(f'train_acc: {train_acc1.item()}%, train_time: {end_train - start_train}s')
        writer.add_scalar('train accuracy top 1', train_acc1.item(), epoch)
        writer.add_scalar('train accuracy top 5', train_acc5.item(), epoch)
        writer.add_scalar('train loss', train_obj.item(), epoch)
        writer.add_scalar('train time', end_train - start_train, epoch)

        # validation
        start_valid = time()
        valid_acc1, valid_acc5, valid_obj = infer(valid_queue, model, criterion)
        scheduler.step()
        end_valid = time()
        # if valid_acc1.item() > best_valid_acc:
        #     best_valid_acc = valid_acc1.item()
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'epoch': epoch,
        #         'valid_acc': valid_acc1.item()
        #     }, os.path.join(model_path, 'best_model_search.pt'))

        print(f'valid_acc: {valid_acc1.item()}%, valid_time: {end_valid - start_valid}s')
        writer.add_scalar('valid accuracy top 1', valid_acc1.item(), epoch)
        writer.add_scalar('valid accuracy top 5', valid_acc5.item(), epoch)
        writer.add_scalar('valid loss', valid_obj.item(), epoch)
        writer.add_scalar('valid time', end_valid - start_valid, epoch)
        # save(model, epoch, optimizer, scheduler, os.path.join(model_path, 'weights_search.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    model.set_tau(
         args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.epochs - 1)
   )

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            print(f'train {step}: loss = {objs.avg.item()}, top 1 = {top1.avg.item()}%, top 5 = {top5.avg.item()}%')

    return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                print(f'valid {step}: loss = {objs.avg.item()}, top 1 = {top1.avg.item()}%, top 5 = {top5.avg.item()}%')

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
