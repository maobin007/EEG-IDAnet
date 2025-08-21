import pylab as p
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.IDAnet import IDAnet
from baseModel import baseModel
import time
import os
import yaml
import random
from dataload.preprocess import get_data
from dataload.dataset import eegDataset
from utils import *
import time

torch.set_num_threads(10)


def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters if p.requires_grad)

def main(config):
    data_path = config['data_path']
    out_folder = config['out_folder']

    lr = config['lr']

    for subId in range(1,10):
        out_path = os.path.join(out_folder, config['network'], 'sub' + str(subId))
        features_path = os.path.join(out_folder, config['network'], 'features', 'sub' + str(subId))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # if not os.path.exists(outpic_path):
        #     os.makedirs(outpic_path)
        if not os.path.exists(features_path):
            os.makedirs(features_path)

        print("Results will be saved in folder: " + out_path)

        dictToYaml(os.path.join(out_path, 'config.yaml'), config)

        setRandom(config['random_seed'])
        train_data, train_labels, test_data, test_labels = get_data(data_path, subject=subId, LOSO=False,
                                                                data_type=config['dataset'], tmin=config['tmin'], tmax=config['tmax'],
                                                                    low_freq=None, high_freq=None)
        train_dataset = eegDataset(train_data, train_labels)
        test_dataset = eegDataset(test_data, test_labels)

        net_args = config['network_args']
        net = eval(config['network'])(**net_args)
        # print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        model = baseModel(net, config, optimizer, loss_func, result_savepath=out_path)

        model.train_test(train_dataset, test_dataset)
        # model.save_tsne_features(test_dataset,results_path=features_path)
        # model.plot_confusion(test_dataset, SubId=subId, results_path=outpic_path, dataset=config['dataset'])


if __name__ == '__main__':
    configFile = 'E:/EEG_IDAnet/config/bciiv2a_transnet.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)

