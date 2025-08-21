import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from dataload.preprocess import get_data
import numpy as np
import matplotlib.pylab as plt

# ==================================模型可解释性——可视化处理——将原始数据、特征以及特征融合、informer分类效果通过t-SNE散点图的形式展现==============================================
def plot_tsne_feature(data_path, feature_path, subject, output_dir, dpi=300, constitute=True, dataset='BCICIV_2a'):
    '''
    Model Interpretability Visulalization
    :param data_path: the path of the data
    :param feature_path: the path of the feature
    :param out_dir: the path of save
    :param constitute: whthere to draw a combination chart(default=True)
    :return:
    '''
    # 图片保存的路径
    output_dir_list = []
    model_type = ["raw", 'DB', 'DB+ECA', 'DB+ECA+Informer']
    for i in range(4):
        output_dir_list.append(output_dir + dataset+ '/sub{:}/'.format(subject)+model_type[i]+'t-sne picture/')
        if not os.path.exists(output_dir_list[i]):
            os.makedirs(output_dir_list[i])
    # 加载数据，获取训练接的标签
    print('Loading data......')
    _, _, _,true_labels = get_data(data_path, subject=subject, LOSO=False, data_type=dataset, tmin=0.,tmax=4., low_freq=None, high_freq=None)
    # 获取所有与特征相关的数据地址，并保存在一个列表
    features_path = []
    features_path.append(os.path.join(feature_path, 'sub{:}'.format(subject), 'raw_features.npy'))
    features_path.append(os.path.join(feature_path, 'sub{:}'.format(subject), 'DB_features.npy'))
    features_path.append(os.path.join(feature_path, 'sub{:}'.format(subject), 'fusion_features.npy'))
    features_path.append(os.path.join(feature_path, 'sub{:}'.format(subject), 'informer_features.npy'))
    # print(features_path)
    data = {}
    for i in range(4):
        features_data = np.load(features_path[i])
        features_data = features_data.reshape(features_data.shape[0], -1)
        print(features_data.shape)
        data[model_type[i]] = features_data
    print('Data loading complete!')

    # 绘图
    print('Image is being generated......')
    if dataset=='BCICIV_2a':
        labels = ['left hand', 'right hand', 'foot', 'tongue']
        colors = [5, 3, 1, 7]
    elif dataset=='BCICIV_2b':
        labels = ['left hand', 'right hand']
        colors = [5, 3]
    else:
        print('无此数据！')
    if constitute:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 10))
        for i in range(4):
            tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, n_iter_without_progress=300, random_state=2024102601)
            X_tsne = tsne.fit_transform(data[model_type[i]])
            X_tsne = MinMaxScaler().fit_transform(X_tsne)

            for category in np.unique(true_labels):
                axs[0, i%3].scatter(
                    *X_tsne[true_labels == category].T,
                    marker = '.',
                    color = plt.cm.Paired(colors[int(category)]),
                    lebel = labels[int(category)],
                    alpha = 0.8,
                    s = 100
                )
            axs[0, i%3].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()
    else:
        tsne = TSNE(n_components=2, perplexity=60, n_iter=3000, n_iter_without_progress=300, random_state=2024102601)
        for i in range(4):
            plt.figure(figsize=(5, 5))
            X_tsne = tsne.fit_transform(data[model_type[i]])
            X_tsne = MinMaxScaler().fit_transform(X_tsne)

            for category in np.unique(true_labels):
                plt.scatter(
                    *X_tsne[true_labels==category].T,
                    marker='.',
                    color=plt.cm.Paired(colors[int(category)]),
                    alpha=0.8,
                    s=100
                )
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.legend()
            output_dir_ = output_dir_list[i] + 'feature_tSNE.png'
            plt.savefig(output_dir_, dpi=dpi)
            print(f'The picture is saved successfully!\nSave address:'+output_dir_)

def attention_score_visualization(attention_path, subject, output_dir, dpi=300):
    # 图片输出路径
    output_dir = output_dir + 's{:}/'.format(subject)+'attention_score_figure/'
    data = np.load(attention_path)
    print(data.shape)
    print(data.min(), data.max())
    for head in range(data.shape[1]): # 5 heads
        for i in range(data.shape[0]//8): # 绘制288个样本的图像，具体绘制多少可以自己设定
            im_cycle = plt.imshow(data[i][head], cmap='RdYlGn', vmax=1., vmin=0.)
            plt.colorbar(im_cycle)
            plt.tight_layout()
            output_dir_ = output_dir + 'headFreq_{:}/attnCycle/'.format(head)
            if not os.path.exists(output_dir_):
                os.makedirs(output_dir_)
            output_dir_Cycle = output_dir_ + "sample_{:}.png".format(i)
            plt.savefig(output_dir_Cycle, dpi=dpi)
            print(f"The picture is saved at {output_dir_Cycle}")
            plt.clf()

def main():
    subject = 3
    data_path = 'E:/data_mb/data/BCICIV_2b/'
    attention_path = r'E:\ProjectEEG_MB\output\bci_iv_2a\wa\mymodel19\attn_weight\sub3\attn_weight.npy'
    # feature_path = 'D:/ProjectEEG_MB/output/bci_iv_2b/wa/mymodel19/features/'
    output_dir = r'E:/ProjectEEG_MB/figures/BCICIV_2a/factor5/'
    # dataset = 'BCICIV_2b'
    # plot_tsne_feature(data_path, feature_path, subject, output_dir, dpi=600, constitute=False, dataset=dataset)
    attention_score_visualization(attention_path, subject, output_dir,dpi=600)


if __name__ == '__main__':
    main()
