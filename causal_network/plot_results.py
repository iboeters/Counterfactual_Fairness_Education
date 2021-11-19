import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.font_manager
import seaborn as sns

def plot_loss(args, train_loss_dict, test_loss_dict):
    # save train and test total loss development
    plt.figure(figsize=(18, 12))
    plt.rcParams["font.family"] = "serif"
    plt.plot(train_loss_dict['train_loss'], label='train', color='green')
    plt.plot(test_loss_dict['index'], test_loss_dict['test_loss'], label='test', color='blue')
    plt.xlabel('Iterations')
    plt.legend()
    plt.title('Loss development training CEVAE')
    plt.tight_layout()
    plt.savefig('output/loss_develop_' + args.filename + '.png')
    plt.savefig('output/loss_develop_' + args.filename + '.pdf')
    plt.close()

    # save train and test loss development of separate loss components
    plt.figure(figsize=(18, 12))
    subindex = 1
    x_axis = 0
    plt.rcParams["font.family"] = "serif"
    for (key, value), (key2, value2) in zip(train_loss_dict.items(), test_loss_dict.items()):
        if key != 'index' and key != 'test_loss':
            plt.subplot(5, 2, subindex)
            plt.plot(train_loss_dict['index'], np.array(value), label='train', color='green')
            plt.plot(test_loss_dict['index'], np.array(value2), label='test', color='blue')
            if key != 'test_loss':
                plt.title(key)
            else:
                plt.title('loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            subindex += 1
    plt.tight_layout()
    plt.legend()
    plt.savefig('output/loss_components_' + args.filename + '.png')
    plt.savefig('output/loss_components_' + args.filename + '.pdf')
    plt.close()

def plot_conf_matrix(df_conf, combination):
    ax = plt.subplot()
    sns.set(font_scale = 0.8)
    fig = sns.heatmap(df_conf, annot=True, annot_kws={"size": 8}, fmt='.2f', ax=ax).get_figure()
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.xaxis.set_ticklabels(['vmbo-b', 'vmbo-k', 'vmbo-g', 'vmbo-t', 'havo', 'vwo'])
    ax.yaxis.set_ticklabels(['vmbo-b', 'vmbo-k', 'vmbo-g', 'vmbo-t', 'havo', 'vwo'])
    fig.savefig("./output/confusion_matrix_" + combination + ".png")
    fig.savefig("./output/confusion_matrix_" + combination + ".pdf")
    plt.clf()

def plot_diff(df, args):
    sns.countplot(data=df, x='diff', color="blue")
    plt.xlabel("Difference in predictor")
    plt.savefig('./output/diff_' + args.filename + '.png')
    plt.savefig('./output/diff_' + args.filename + '.pdf')
    plt.clf()
