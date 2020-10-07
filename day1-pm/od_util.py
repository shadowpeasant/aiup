import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

samples = 5

def feature_score(X_orig, X_recon):
    fscore = np.power(X_orig - X_recon, 2)
    fscore = fscore.reshape((-1, samples) + X_orig.shape[1:])
    fscore = np.mean(fscore, axis=1)
    print('shape of X_orig= {}, shape of fscore={}'.format(X_orig.shape, fscore.shape))
    return fscore
    
def instance_score(fscore, outlier_perc = 100.):
    fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
    print('shape of fscore_flat={}'.format(fscore_flat.shape))
    n_score_features = int(np.ceil(.01 * outlier_perc * fscore_flat.shape[1]))
    sorted_fscore = np.sort(fscore_flat, axis=1)
    sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
    iscore = np.mean(sorted_fscore_perc, axis=1)
    print('shape of iscore={}'.format(iscore.shape))
    return iscore


def score(model, X, outlier_perc = 100.): 

    # reconstruct instances
    X_samples = np.repeat(X, samples, axis=0)
    X_recon = model.predict(X_samples)

    # compute feature and instance level scores
    fscore = feature_score(X_samples, X_recon)
    iscore = instance_score(fscore, outlier_perc=outlier_perc)

    return fscore, iscore

def infer_threshold(model, X, outlier_type = 'instance',outlier_perc = 100.,threshold_perc = 95.):
    
    # compute outlier scores
    fscore, iscore = score(model, X, outlier_perc=outlier_perc)
    if outlier_type == 'feature':
        outlier_score = fscore
    elif outlier_type == 'instance':
        outlier_score = iscore
    else:
        raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

    # update threshold
    threshold = np.percentile(outlier_score, threshold_perc)
    return threshold

def plot_instance_score(preds, target, labels, threshold, ylim = (None, None)):
    
    scores = preds['data']['instance_score']
    df = pd.DataFrame(dict(idx=np.arange(len(scores)), score=scores, label=target))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.idx, group.score, marker='o', linestyle='', ms=6, label=labels[name])
    plt.plot(np.arange(len(scores)), np.ones(len(scores)) * threshold, color='g', label='Threshold')
    plt.ylim(ylim)
    plt.xlabel('Number of Instances')
    plt.ylabel('Instance Level Score')
    ax.legend()
    plt.show()

def plot_feature_outlier_tabular(od_preds, 
                                 X, 
                                 X_recon = None, 
                                 threshold = None,
                                 instance_ids=None,
                                 max_instances = 5,
                                 top_n = int(1e12),
                                 outliers_only = False,
                                 feature_names = None,
                                 width = .2,
                                 figsize = (20, 10)):
    if outliers_only and instance_ids is None:
        instance_ids = list(np.where(od_preds['data']['is_outlier'])[0])
    elif instance_ids is None:
        instance_ids = list(range(len(od_preds['data']['is_outlier'])))
    n_instances = min(max_instances, len(instance_ids))
    instance_ids = instance_ids[:n_instances]
    n_features = X.shape[1]
    n_cols = 2

    labels_values = ['Original']
    if X_recon is not None:
        labels_values += ['Reconstructed']
    labels_scores = ['Outlier Score']
    if threshold is not None:
        labels_scores = ['Threshold'] + labels_scores

    fig, axes = plt.subplots(nrows=n_instances, ncols=n_cols, figsize=figsize)

    n_subplot = 1
    for i in range(n_instances):

        idx = instance_ids[i]

        fscore = od_preds['data']['feature_score'][idx]
        if top_n >= n_features:
            keep_cols = np.arange(n_features)
        else:
            keep_cols = np.argsort(fscore)[::-1][:top_n]
        fscore = fscore[keep_cols]
        X_idx = X[idx][keep_cols]
        ticks = np.arange(len(keep_cols))

        plt.subplot(n_instances, n_cols, n_subplot)
        if X_recon is not None:
            X_recon_idx = X_recon[idx][keep_cols]
            plt.bar(ticks - width, X_idx, width=width, color='b', align='center')
            plt.bar(ticks, X_recon_idx, width=width, color='g', align='center')
        else:
            plt.bar(ticks, X_idx, width=width, color='b', align='center')
        if feature_names is not None:
            plt.xticks(ticks=ticks, labels=list(np.array(feature_names)[keep_cols]), rotation=45)
        plt.title('Feature Values')
        plt.xlabel('Features')
        plt.ylabel('Feature Values')
        plt.legend(labels_values)
        n_subplot += 1

        plt.subplot(n_instances, n_cols, n_subplot)
        plt.bar(ticks, fscore)
        if threshold is not None:
            plt.plot(np.ones(len(ticks)) * threshold, 'r')
        if feature_names is not None:
            plt.xticks(ticks=ticks, labels=list(np.array(feature_names)[keep_cols]), rotation=45)
        plt.title('Feature Level Outlier Score')
        plt.xlabel('Features')
        plt.ylabel('Outlier Score')
        plt.legend(labels_scores)
        n_subplot += 1

    plt.tight_layout()
    plt.show()