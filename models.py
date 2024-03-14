import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones


class TransferNet(nn.Module):
    def __init__(self, num_class,
                 base_net='alexnet',
                 transfer_loss='mmd',
                 use_bottleneck=False,
                 bottleneck_width=256,
                 max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        if feature_dim != num_class:
            classifier_list = [
                # nn.Linear(feature_dim, num_class),
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_class)
                ]
            self.classifier_layer = nn.Sequential(*classifier_list)
        self.transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.adapt_loss = TransferLoss(**self.transfer_loss_args)

    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)
        elif self.transfer_loss == 'LWD':
            source_clf = self.classifier_layer(source)
            wd_weights = self.classifier_layer(target)
            kwargs['source_label'] = source_label
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            kwargs['wd_weights'] = wd_weights
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)

        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss

    
    def get_parameters(self, initial_lr=1.0):
        # params = [
        #     {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
        #     {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        # ]
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def pretrain(self, data, label):
        features = self.base_network(data)
        source = self.classifier_layer(features)
        clf_loss = self.criterion(source, label)

        return clf_loss, source

    def predict(self, x):
        x = self.base_network(x)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

    def output_feature(self, x):
        feature = self.base_network(x)
        return feature