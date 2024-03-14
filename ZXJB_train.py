import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import sys
import pandas as pd
# import heartrate


# heartrate.trace(browser=True)

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)

    # network related
    parser.add_argument('--backbone', type=str, default='alexnet2d')
    parser.add_argument('--use_bottleneck', type=str2bool, default=False)

    # data loading related
    parser.add_argument('--data_dir', type=str, default=r'D:\DATABASE\ZXJ_GD\sample')
    parser.add_argument('--src_domain', type=str, default="10-B1-4_CH1-8")
    parser.add_argument('--tgt_domain', type=str, default="10-B1-4_CH1-8")
    parser.add_argument('--num_class', type=int, default=8)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--graph', type=bool, default=False)
    parser.add_argument('--data_type', type=str, default="orig_sample")

    # training related
    parser.add_argument('--pretrain', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.data_type, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.data_type, args.tgt_domain)
    source_loader = data_loader.load_data(
        folder_src, args.batch_size, train=True, ratio=args.train_ratio, graph=args.graph)
    source_test_loader = data_loader.load_data(
        folder_src, args.batch_size, train=False, ratio=args.train_ratio, graph=args.graph)
    target_train_loader = data_loader.load_data(
        folder_tgt, args.batch_size, train=True, ratio=args.train_ratio, graph=args.graph)
    target_test_loader = data_loader.load_data(
        folder_tgt, args.batch_size, train=False, ratio=args.train_ratio, graph=args.graph)
    return source_loader, source_test_loader, target_train_loader, target_test_loader


def get_model(args):
    model = models.TransferNet(num_class=args.num_class,
                               transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
                               use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


def pre_train(model, source_loader, source_test_loader,  optimizer, lr_scheduler, args):
    print('pretrain,domain:', args.src_domain)
    len_source_loader = len(source_loader)
    n_batch = args.batch_size
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
    for repeat_n in range(0, 3):
        best_acc = 0
        stop = 0
        log = []
        for e in range(1, args.n_epoch + 1):
            model.train()
            train_loss_clf = utils.AverageMeter()
            model.epoch_based_processing(n_batch)
            if len_source_loader != 0:
                iter_source = iter(source_loader)

            for f in range(len_source_loader):
                data_source, label_source, _ = next(iter_source)  # .next()
                data_source, label_source = data_source.to(
                    args.device), label_source.to(args.device)

                clf_loss, predict = model.pretrain(data_source, label_source)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()

                train_loss_clf.update(clf_loss.item())

            log.append(train_loss_clf.avg)

            info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}'.format(
                e, args.n_epoch, train_loss_clf.avg)
            # Test
            stop += 1
            test_acc, test_loss = test(model, source_test_loader, args)
            info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
            np_log = np.array(log, dtype=float)
            np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
            if best_acc < test_acc:
                best_acc = test_acc
                stop = 0
            if best_acc > 95:
                print('Saving checkpoints', '/domain:', args.src_domain)
                ckpt = {'state_dict': model.state_dict()}
                pretrain_dir = r'D:\\python_workfile\\TL-comparsion\\Deep\\pretrain\\' + args.backbone + '\\' + args.data_type + '\\'
                if os.path.exists(pretrain_dir):
                    torch.save(ckpt, pretrain_dir + args.src_domain + '.tar')
                else:
                    os.makedirs(pretrain_dir)
                    torch.save(ckpt, pretrain_dir + args.src_domain + '.tar')
            if args.early_stop > 0 and stop >= args.early_stop:
                print(info)
                break
            print(info)
        print('Pretrain result: {:.4f}'.format(best_acc), '\nRepeat num:', repeat_n)
        if best_acc >= 95:
            break
    return best_acc


def test(model, test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_test_dataset = len(test_loader) * args.batch_size
    if args.pretrain is False:
        with torch.no_grad():
            for data in test_loader:
                data, target = data[0].to(args.device), data[1].to(args.device)
                s_output = model.predict(data)
                loss = criterion(s_output, target)
                test_loss.update(loss.item())
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target)
        acc = 100. * correct / len_test_dataset
    else:
        with torch.no_grad():
            for data in test_loader:
                data, target = data[0].to(args.device), data[1].to(args.device)
                loss, s_output = model.pretrain(data, target)
                test_loss.update(loss.item())
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target)
        acc = 100. * correct / len_test_dataset

    return acc, test_loss.avg


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    if args.pretrain is False:
        ckpt = torch.load(r'/root/autodl-tmp/project/Deep_DA/pretrain/ZXJB/'
                          + args.backbone + '/' + args.data_type + '/' + args.src_domain + '.tar')
        model.load_state_dict(ckpt["state_dict"], strict=False)
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []

    orig_test_acc, _ = test(model, target_test_loader, args)

    for e in range(1, args.n_epoch + 1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for _ in range(n_batch):
            if args.graph:
                data_source, label_source = next(iter_source)
                data_target, _ = next(iter_target)
            else:
                data_source, label_source, _ = next(iter_source)  # .next()
                data_target, _, _ = next(iter_target)  # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)

            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    print('Transfer result: {:.4f}'.format(best_acc))
    return orig_test_acc, best_acc


def main():
    # 训练模式，orig_sample 原始数据训练， resampled 重采样处理数据训练， freq_domain频域数据训练， t-f时频域图片数据
    data_type_list = ['orig_sample', 'resampled', 'freq_sample', 'time-freq', 'envo']
    data_type = data_type_list[0]
    # Transfer_task指的是不同的迁移任务： 相同速度不同设备same_speed, 相同设备不同速度：same_equip, 不同转速 var_speed
    Transfer_task = 'hybrid'
    Attention = False
    # 模型参数目录
    config_dir = r'./YAML'
    # data_dir = r'/root/autodl-tmp/sample/smple_DeepDADG'
    data_dir = r'D:\DATABASE\ZXJ_test_data\fault_bearing_standard_sample\smple_DeepDADG'
    # model_list = os.listdir(config_dir)
    model_list = ['LWD.yaml']
    domain_list = os.listdir(data_dir+'/'+data_type)

    for model_config in model_list:
        sys.argv[1] = '--config'
        sys.argv[2] = config_dir + '/' + model_config
        info_list = []
        for i in domain_list:
            # 根据迁移的方式 确定迁移训练的目标域和源域
            for j in domain_list:
                if Transfer_task == 'same_speed':
                    if i.split('_')[1] != j.split('_')[1]:
                        continue
                elif Transfer_task == 'same_equip':
                    if i.split('_')[2] != j.split('_')[2]:
                        continue
                elif Transfer_task == 'var_speed':
                    if i.split('_')[1] == j.split('_')[1] or i.split('_')[2] != j.split('_')[2]:
                        continue
                if i == j:
                    continue

                # 获得训练参数
                parser = get_parser()
                args = parser.parse_args()
                setattr(args, "data_type", data_type)
                setattr(args, "backbone", 'alexnet')

                # 如果使用时频图训练，则需要更换所需要的数据处理方式和backbone
                if data_type == 'time-freq':
                    setattr(args, "graph", True)
                    setattr(args, "backbone", args.backbone + '_2d')
                # 是否使用注意力机制
                if Attention:
                    setattr(args, "backbone", args.backbone + '_attention')
                # 如果/pretrain 目录下无预训练权重则进行预训练
                if os.path.exists(
                         './pretrain/ZXJB/'
                        + args.backbone + '/' + args.data_type + '/'
                        + i + '.tar'):
                    setattr(args, "pretrain", False)
                else:
                    setattr(args, "pretrain", True)

                setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                setattr(args, "src_domain", i)
                setattr(args, "tgt_domain", j)
                setattr(args, "data_dir", data_dir)
                setattr(args, "data_type", data_type)
                set_random_seed(args.seed)
                source_loader, source_test_loader, target_train_loader, target_test_loader = load_data(args)
                if args.epoch_based_training:
                    setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
                else:
                    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
                print(args)

                # 构建网络、优化器等结构
                model = get_model(args)
                optimizer = get_optimizer(model, args)
                if args.lr_scheduler:
                    scheduler = get_scheduler(optimizer, args)
                else:
                    scheduler = None
                # 模型预训练
                if args.pretrain is True:
                    best_acc = pre_train(model, source_loader, source_test_loader, optimizer, scheduler, args)
                    if best_acc >= 95:
                        print('Model pretrain complete. Best acc:', best_acc)
                    elif best_acc < 95:
                        print('Model pretrain failed:', i)
                        break
                # 迁移训练
                orig_acc, transfer_result = train(source_loader, target_train_loader, target_test_loader, model,
                                                  optimizer, scheduler, args)
                info_list.append([args.data_type, model_config, i, j, float(orig_acc), float(transfer_result)])
        info = pd.DataFrame(info_list,
                            columns=['Data-type', 'Model', 'Source', 'Target', 'Original Test Acc', 'Transfer Acc'])
        # 保存训练结果记录
        record_path = r'./Record/' + args.backbone + '/' + Transfer_task + '/'
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        info.to_excel(record_path + data_type + '-' + model_config[:-4] + '.xlsx', index=False)


if __name__ == "__main__":
    main()
