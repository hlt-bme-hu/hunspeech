import os
import sys
import logging

import numpy as np
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GMM
from features import mfcc

def shifted_delta_cepstra(wav_fn, delta=1, shift=3, k_conc=3):
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


def get_sdc_feats():
    data_fn = '/mnt/store/makrai/data/speech/jewel/jewel-shifted.npy'
    if os.path.isfile(data_fn):
        return np.load(open(data_fn))
    else:
        wav_dir = '/mnt/store/hlt/Speech/Jewels/wav/'
        data = np.concatenate([shifted_delta_cepstra('{}/{}'.format(wav_dir, wav_fn))
                               for wav_fn in os.listdir(wav_dir)])
        np.save(open('jewel-shifted.npy', mode='w'), data)
        return data


def cluster(data, n_clusters=6):
    classers = {
        "KMeans++": KMeans(n_clusters=n_clusters, init='k-means++', n_jobs=4),
        "KMeans-rand": KMeans(n_clusters=n_clusters, init='random', n_jobs=4),
        # "TODO": TODO KMeans(n_clusters=n_clusters, init='PCA', n_jobs=4)
        "AffinityPropagation": AffinityPropagation(),
        "MeanShift": MeanShift(),#n_jobs=4),
        "SpectralClustering": SpectralClustering(n_clusters=n_clusters),
        # "AgglomerativeClustering": AgglomerativeClustering:
        #    ""linkage": linkage" determines the metric used for the merge strategy
        "AgglomerativeClustering-ward": AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
        "AgglomerativeClustering-compl": AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
        "AgglomerativeClustering-avg": AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
        "DBSCAN": DBSCAN(),
        "GMM": GMM(n_components=1) ,
        # "covariance": covariance_type='spherical', 'tied', 'diag', 'full
        "Birch": Birch(n_clusters=n_clusters)
    }
    for clser_name, classer in classers.iteritems():
        dump_fn = '/mnt/store/makrai/data/speech/jewel/{}.npy'.format(clser_name)
        if os.path.isfile(dump_fn):
            continue
        logging.info(clser_name)
        classer.fit(data)
        classes = classer.predict(data)
        np.save(dump_fn, classes)

if __name__ == "__main__":
    data = get_sdc_feats()
    cluster(data)
