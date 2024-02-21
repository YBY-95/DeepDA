import torch
import models
import configargparse
from main import get_parser
import data_loader
import os
import sys
import pandas as pd
import numpy as np


def load_data(args):
    folder_src = os.path.join(args.data_dir, args.data_type, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.data_type, args.tgt_domain)
    source_loader = data_loader.load_data(
        folder_src, args.batch_size, train=True, ratio=args.train_ratio, graph=args.graph)
    target_train_loader = data_loader.load_data(
        folder_tgt, args.batch_size, train=True, ratio=args.train_ratio, graph=args.graph)
    return source_loader, target_train_loader


def get_feature(source_loader, target_loader, model, args):
    ckpt = torch.load(r'D:\\python_workfile\\TL-comparsion\\Deep\\pretrain\\'
                      + args.backbone + '\\' + args.data_type + '\\' + args.src_domain + '.tar')
    model.load_state_dict(ckpt["state_dict"], strict=False)
    m = 0
    n = 0
    model.eval()

    for data in source_loader:
        data, target = data[0].to(args.device), data[1].to(args.device)
        source_feature = model.output_feature(data)
        source_feature = source_feature.cpu()
        source_feature = source_feature.detach().numpy()
        if m == 0:
            source_data = source_feature
        else:
            source_data = np.vstack((source_data, source_feature))
        if m == 10:
            source_data = source_data.reshape([source_data.size, 1])
            break
        m += 1

    for data in target_loader:
        data, target = data[0].to(args.device), data[1].to(args.device)
        target_feature = model.output_feature(data)
        target_feature = target_feature.cpu()
        target_feature = target_feature.detach().numpy()
        if n == 0:
            target_data = target_feature
        else:
            target_data = np.vstack((target_data, target_feature))

        if n == 10:
            target_data = target_data.reshape([target_data.size, 1])
            break
        n += 1

    return source_data, target_data


def get_model(args):
    model = models.TransferNet(num_class=args.num_class,
                               transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
                               use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def set_parser(data_type, backbone, attention, source, target):
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "data_type", data_type)
    setattr(args, "backbone", backbone)
    setattr(args, "src_domain", source)
    setattr(args, "tgt_domain", target)
    setattr(args, "max_iter", 10000)
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if attention:
        setattr(args, "backbone", args.backbone + '_attention')

    return args


def main():
    # 这里输入的是原始数据orig，输出的是特征feature
    data_type = 'envo'
    backbone = 'alexnet'
    attention = False
    speed_list = ['10', '50', '100', '150', '200', '250', '300', '350']
    equip_list = ['B1-4_CH1-8', 'B5-8_CH9-16']

    config_dir = r'D:\python_workfile\TL-comparsion\Deep\YAML\BNM.yaml'
    feature_dir = r'D:\DATABASE\ZXJ_GD\feature' + "/" + data_type + "/" + backbone
    sys.argv[1] = '--config'
    sys.argv[2] = config_dir

    equip = equip_list[0]
    for source_speed in speed_list:
        for target_speed in speed_list:
            if source_speed == target_speed:
                continue
            source_name = source_speed + '-' + equip
            target_name = target_speed + '-' + equip

            args = set_parser(data_type, backbone, attention, source_name, target_name)

            source_loader, target_loader = load_data(args)

            model = get_model(args)

            source_feature, target_feature = get_feature(source_loader, target_loader, model, args)

            save_dir = feature_dir + '/' + source_speed + '/' + equip
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            pd.DataFrame(target_feature).to_csv(save_dir + '/' + target_speed + '.csv')

            print(source_speed + '-' + target_speed)

        pd.DataFrame(source_feature).to_csv(save_dir + '/' + source_speed + '.csv')
if __name__ == '__main__':
    main()
