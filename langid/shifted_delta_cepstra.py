import logging
import os
import pickle

import numpy as np
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GMM
from features import mfcc

# TODO
"""
AffinityPropagation and SpectralClustering take similarity matrices.  These
can be obtained from the functions in the sklearn.metrics.pairwise module
"""

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #if log_to_err
    handler = logging.StreamHandler()
    # else log_fn = "~/l" if os.path.isfile(log_fn): os.remove(log_fn) handler
    # = logging.FileHandler(log_fn)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


class ShiftedDeltaClusterer():
    def __init__(self,n_clusters=6, n_jobs=8):
        self.wav_dir = '/mnt/store/hlt/Speech/Jewels/wav/'
        self.project_dir = '/mnt/store/makrai/data/speech/jewel'
        self.algos = [
            ("KMeans++", KMeans(n_clusters=n_clusters, init='k-means++', n_jobs=n_jobs)),
            ("KMeans-rand", KMeans(n_clusters=n_clusters, init='random', n_jobs=n_jobs)),
            # "TODO", TODO KMeans(n_clusters=n_clusters, init='PCA', n_jobs=n_jobs)
            ("AffinityPropagation", AffinityPropagation()),
            ("MeanShift", MeanShift()),
            ("SpectralClustering", SpectralClustering(n_clusters=n_clusters)),
            # AgglomerativeClustering:
            #     "linkage" determines the metric used for the merge strategy
            ("AgglomerativeClustering-ward", AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')),
            ("AgglomerativeClustering-compl", AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')),
            ("AgglomerativeClustering-avg", AgglomerativeClustering(n_clusters=n_clusters, linkage='average')),
            ("DBSCAN", DBSCAN()),
            ("GMM", GMM(n_components=1)),
            # "covariance", covariance_type=
            #    'spherical', 'tied', 'diag', 'full'
            ("Birch", Birch(n_clusters=n_clusters)) # TODO n_clusters kell neki?
        ]

    def get_sdc_all_tracks(self):
        data_fn = '{}/{}'.format(self.project_dir, 'sdc_all_jewel.npy')
        if os.path.isfile(data_fn):
            self.sdc_all_speech = np.load(open(data_fn))
        else:
            logger.info(
                'Computing shifted delta cepstra for all speech in {}'.format(
                    self.wav_dir))
            self.sdc_all_speech = np.concatenate([self.shifted_delta_cepstra(
                '{}/{}'.format( self.wav_dir, wav_fn))
                for wav_fn in os.listdir(self.wav_dir)])
            np.save(open('sdc_all_jewel.npy', mode='w'), self.sdc_all_speech)

    def shifted_delta_cepstra(self, wav_fn, delta=1, shift=3, k_conc=3):
        """
        :param delta: represents the time advance and delay for the delta computation,
        :param k_conc: is the number of blocks whose delta coefficients are concatenated
        :param shift: is the time shift between consecutive blocks
        See the paper
            PA Torres-Carrasquillo et al (2002)
            Approaches to language identification using Gaussian mixture models
                and shifted delta cepstral features.
        """
        (rate,sig) = wav.read(wav_fn)
        features = mfcc(sig,rate)
        features = features[delta:] - features[:-delta]
        shifted = np.zeros((features.shape[0]-shift*k_conc, features.shape[1]))
        for i in xrange(shifted.shape[0]):
            shifted[i] = features[i:i+k_conc:shift]
        return shifted

    def cluster(self):
        cluster_dir = '{}/cluster'.format(self.project_dir)
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)
        for algo_name, algo in self.algos:
            algo_dir = '{}/{}'.format(cluster_dir, algo_name)
            classer = self.get_classer(algo_name, algo, algo_dir)
            for wav_fn in os.listdir(self.wav_dir):
                track_to_clust_fn = '{}/{}.npy'.format(
                    algo_dir, os.path.splitext(wav_fn)[0])
                if os.path.isfile(track_to_clust_fn):
                    continue
                logger.info('Assigning {} by {}'.format(wav_fn,algo_name))
                assign = classer.predict(self.shifted_delta_cepstra(
                    '{}/{}'.format(self.wav_dir, wav_fn)))
                np.savetxt(track_to_clust_fn, assign)

    def get_classer(self, algo_name, classer, algo_dir):
        if not os.path.exists(algo_dir):
            os.mkdir(algo_dir)
        dump_fn = '{}/{}.npy'.format(algo_dir, algo_name)
        if os.path.isfile(dump_fn):
            return pickle.load(open(dump_fn, mode='rb'))
        logger.info('clustering all speech with {}'.format(algo_name))
        classer.fit(self.sdc_all_speech)
        logger.info('dumping classifier')
        pickle.dump(classer, open(dump_fn, mode='wb'))
        return classer

    def main(self):
        self.get_sdc_all_tracks()
        self.cluster()


if __name__ == "__main__":
    logger = get_logger()
    ShiftedDeltaClusterer().main()
