import os 
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

root_logdir = os.path.join(os.curdir, "tb_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

    
def draw_anomaly(y_true, error, threshold):
        groupsDF = pd.DataFrame({'error': error,
                                 'true': y_true}).groupby('true')
        print('y counts', y_true.value_counts())
        figure, axes = plt.subplots(figsize=(12, 8))
        axes.set_ylim(0,30)
        for name, group in groupsDF:
            axes.plot(group.index, group.error, marker='x' if name == 1 else 'o', linestyle='',
                    color='r' if name == 1 else 'g', label="Anomaly" if name == 1 else "Normal")

        axes.hlines(threshold, axes.get_xlim()[0], axes.get_xlim()[1], colors="b", zorder=100, label='Threshold')
        axes.legend()
        
        plt.title("Anomalies")
        plt.ylabel("Error")
        plt.xlabel("Data")
        plt.show()
        
K = tf.keras.backend
kl_divergence = tf.keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
        
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))


    def infer_threshold(self,
                        X: np.ndarray,
                        outlier_type: str = 'instance',
                        outlier_perc: float = 100.,
                        threshold_perc: float = 95.,
                        batch_size: int = int(1e10)
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        outliers in a sample of the dataset.
        Parameters
        ----------
        X
            Batch of instances.
        outlier_type
            Predict outliers at the 'feature' or 'instance' level.
        outlier_perc
            Percentage of sorted feature level outlier scores used to predict instance level outlier.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        batch_size
            Batch size used when making predictions with the autoencoder.
        """
        # compute outlier scores
        fscore, iscore = self.score(X, outlier_perc=outlier_perc, batch_size=batch_size)
        if outlier_type == 'feature':
            outlier_score = fscore
        elif outlier_type == 'instance':
            outlier_score = iscore
        else:
            raise ValueError('`outlier_score` needs to be either `feature` or `instance`.')

        # update threshold
        threshold = np.percentile(outlier_score, threshold_perc)
        return threshold
    
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 