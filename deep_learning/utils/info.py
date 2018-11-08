import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import os
import json
import argparse
import pprint as p

def get_models(model_dir, count=None):
    if count:
        return os.listdir(model_dir)[:count]
    else:
        return os.listdir(model_dir)
def get_accuracy(model):
    acc = []
    for f in range(1, len(os.listdir(model))):
        try:
            with open(os.path.join(model, 'accuracy_{}.json'.format(f)), 'r') as m:
                acc.append(json.load(m)['acc'])
        except:
            continue
    return acc

def get_loss(model):
    loss = []
    for f in range(1, len(os.listdir(model))):
        try:
            with open(os.path.join(model, 'accuracy_{}.json'.format(f)), 'r') as m:
                loss.append(json.load(m)['test_loss'])
        except:
            continue
    return loss

def get_eer(model):
    EER = []
    for f in range(1, len(os.listdir(model))):
        try:
            with open(os.path.join(model, 'accuracy_{}.json'.format(f)), 'r') as m:
                EER.append(json.load(m)['EER'])
        except:
            continue
    return EER

def get_precision(model):
    precision = []        
    for f in range(1, len(os.listdir(model))):
        try:
            with open(os.path.join(model, 'accuracy_{}.json'.format(f)), 'r') as m:
                precision.append(json.load(m)['precision'])
        except:
            continue
    return precision

def get_recall(model):
    recall = []
    for f in range(1, len(os.listdir(model))):
        try:
            with open(os.path.join(model, 'accuracy_{}.json'.format(f)), 'r') as m:
                recall.append(json.load(m)['recall'])
        except:
            continue
    return recall

def get_accuracies(models):
    acc = []
    for model in models:
        try:
            with open(os.path.join(model, 'best_model.json'), 'r') as m:
                acc.append(json.load(m)['acc'])
        except:
            continue
    return acc

def show_confusion_matrix(model):
    cm = None
    with open(os.path.join(model, 'best_model.json'), 'r') as m:
        cm = json.load(m)['confusion']
        print(cm)
    fig = sn.heatmap(cm, annot=True, fmt='.0f',  cmap="YlGnBu")
    fig.set_ylabel('True Label')
    fig.set_xlabel('Predicted')

    plt.show()
    return cm

def show_normalized_cm(model, jfile = 'best_model.json'):
    cm = None
    with open(os.path.join(model, jfile), 'r') as m:
        cm = json.load(m)['confusion']
    
    for i in range(len(cm)):
        total = sum(cm[i])
        for j in range(len(cm[i])):
            cm[i][j] = cm[i][j] / total
    fig = sn.heatmap(cm, annot=True, fmt='.4f',  cmap="YlGnBu")
    fig.set_title('Normalized Confusion Matrix')
    fig.set_ylabel('True Label')
    fig.set_xlabel('Predicted')

    plt.show()
    return cm

def show_acc_epoch(accs):
    plt.plot(accs)
    plt.title('accuracy vs epoch')
    plt.show()

def show_loss_epoch(loss):
    plt.plot(loss)
    plt.title('loss vs epoch')
    plt.show()

def show_fscores(loss):
    plt.plot(loss)
    plt.title('average fscores vs epoch')
    plt.show()

def print_info(model):
    with open(model, 'r') as f:
        p.pprint(json.loads(f))

def avg_fscore(model):
    f = []
    for metadata in range(1, len(os.listdir(model))):
        try:
            with open(os.path.join(model, 'accuracy_{}.json'.format(metadata)), 'r') as m:
                scores = json.load(m)['f-score']
                f.append(sum(scores.values())/len(scores))
        except:
            continue
    return f