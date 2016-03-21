import logging
from math import log
import os
import pickle

import numpy as np
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, estimate_bandwidth
from sklearn.mixture import GMM
from sklearn.manifold import TSNE
from features import mfcc


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


class ShiftedDeltaClusterer():
    def __init__(self,n_clusters=6, n_jobs=-1):
        """
        Cluster speech frames by language, based on shifted delta cepstral
        features.  Many clustering algorithms by sklearn are tried.

        AffinityPropagation 
            not used, because its time complexity is quadratic in the number
            of samples.
        MeanShift
            idea
                the mean shift vector is computed for each centroid and points
                towards a region of the maximum increase in the density 
            complexity 
                O(T*n*log(n)) in lower dimensions, with n the number of
                samples and T the number of points. In higher dimensions the
                complexity will tend towards O(T*n^2).  
            Scalability can be boosted by using fewer seeds, 
                for example by using a higher value of min_bin_freq in the
                get_bin_seeds function.  
            Note that the estimate_bandwidth function is much less scalable
                than the mean shift algorithm and will be the bottleneck if it
                is used.
        AgglomerativeClustering:
            "linkage" determines the metric used for the merge strategy
            scalability
                scalable, when when it is used jointly with a connectivity
                matrix, but is computationally expensive when no connectivity
                constraints are added between samples: it considers at each
                step all the possible merges.
        TSNE
            perplexity http://lvdmaaten.github.io/tsne/
        """
        homes = '/home' if True else '/mnt/store'
        self.wav_dir = os.path.join(homes, 'hlt/Speech/Jewels/wav/')
        self.project_dir = os.path.join(homes,
                                        'makrai/data/unsorted/speech/jewel')
        n_comp = 1 # GMM
        self.algos = [
            ("KMeans++", KMeans(n_clusters=n_clusters, init='k-means++', n_jobs=n_jobs)),
            ("KMeans-rand", KMeans(n_clusters=n_clusters, init='random', n_jobs=n_jobs)),
            # TODO try preprocessing by PCA before KMeans
            ("MeanShift", MeanShift(n_jobs=n_jobs, bandwidth=56.3255)), 
            #   bandwidth estimated from points :16384 TODO
            ("SpectralClustering", SpectralClustering(n_clusters=n_clusters)),
            # TODO
            #   ("AgglomerativeClustering-ward", AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')),
            #   ("AgglomerativeClustering-compl", AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')),
            #   ("AgglomerativeClustering-avg", AgglomerativeClustering(n_clusters=n_clusters, linkage='average')),
            ("DBSCAN", DBSCAN()),
            #   algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            #       The algorithm to be used to find nearest neighbors
            #   has no attribute 'predict but has fit_predict
            ("GMM-spherical", GMM(n_components=n_comp, covariance_type='spherical')),
            ("GMM-tied", GMM(n_components=n_comp, covariance_type='tied')),
            ("GMM-diag", GMM(n_components=n_comp, covariance_type='diag')),
            ("GMM-full", GMM(n_components=n_comp, covariance_type='full')),
            # ("Birch", Birch(n_clusters=n_clusters)), # TODO
            #   n_clusters kell neki?
            #   MemoryError
            #("TSNE", TSNE())
        ]

    def get_sdc_all_tracks(self):
        data_fn = os.path.join(self.project_dir, 'sdc_all_jewel.npy')
        if os.path.isfile(data_fn):
            self.sdc_all_speech = np.load(open(data_fn))
        else:
            logger.info(
                'Computing shifted delta cepstra for all speech in {}'.format(
                    self.wav_dir))
            self.sdc_all_speech = np.concatenate([self.shifted_delta_cepstra(
                os.path.join(self.wav_dir, wav_fn))
                for wav_fn in os.listdir(self.wav_dir)])
            np.save(open(data_fn, mode='w'), self.sdc_all_speech)

    def shifted_delta_cepstra(self, wav_fn, delta=1, shift=3, k_conc=3):
        """
        :param delta: represents the time advance and delay for the delta computation,
        :param k_conc: is the number of blocks whose delta coefficients are concatenated
        :param shift: is the time shift between consecutive blocks

        Shifted delta cepstra are feature vectors created by stacking delta
        cepstra computed across multiple speech frames.
        See the paper
            PA Torres-Carrasquillo et al (2002)
            Approaches to language identification using Gaussian mixture models
                and shifted delta cepstral features.

        len(mfcc) == 39 == 3 * (12 cepstral + 1 energy)
        """
        (rate,sig) = wav.read(wav_fn)
        features = mfcc(sig,rate)
        logger.debug('Cepstral data of shape {} read from {}'.format(
            features.shape, wav_fn))
        # TODO include original cepstra as well?
        features = features[delta:] - features[:-delta]
        shifted = np.zeros((features.shape[0]-shift*k_conc, 
                            k_conc * features.shape[1]))
        for i in xrange(shifted.shape[0]):
            shifted[i,:] = features[i:i+k_conc*shift:shift,:].reshape((1,-1))
        return shifted

    def cluster(self):
        cluster_dir = '{}/cluster'.format(self.project_dir)
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)
        for algo_name, algo in self.algos:
            algo_dir = os.path.join(cluster_dir, algo_name)
            classer = self.get_classer(algo_name, algo, algo_dir)
            for wav_fn in os.listdir(self.wav_dir):
                track_to_clust_fn = '{}.npy'.format(
                    os.path.join(algo_dir, os.path.splitext(wav_fn)[0]))
                if os.path.isfile(track_to_clust_fn):
                    continue
                logger.info('Assigning {} by {}'.format(wav_fn,algo_name))
                #if hasattr(classer, 'predict')
                assign = classer.predict(self.shifted_delta_cepstra(
                    os.path.join(self.wav_dir, wav_fn)))
                np.savetxt(track_to_clust_fn, assign, fmt='%i')

    def get_classer(self, algo_name, classer, algo_dir):
        if not os.path.exists(algo_dir):
            os.mkdir(algo_dir)
        classer_fn = '{}_classer.npy'.format(os.path.join(algo_dir, algo_name))
        trafoed_fn = '{}_trafoed.npy'.format(os.path.join(algo_dir, algo_name))
        if os.path.isfile(classer_fn):
            return pickle.load(open(classer_fn, mode='rb'))
        else:
            logger.info('clustering all speech with {}'.format(algo_name))
            if False and algo_name == 'MeanShift':
                len_ = 4
                while  len_ < self.sdc_all_speech.shape[0]:
                    logging.info((len_,
                                  estimate_bandwidth(self.sdc_all_speech[:len_])))
                    len_ *= 2
            elif hasattr(classer, 'fit_transform'):
                all_speech_trafoed = classer.fit_transform(self.sdc_all_speech)
                np.save(open(trafoed_fn, mode='wb'), all_speech_trafoed)
            elif hasattr(classer, 'fit'):
                classer.fit(self.sdc_all_speech)
            else:
                classer.fit_predict(self.sdc_all_speech)
            logger.info(classer.get_params())
            logger.info('dumping classifier')
            pickle.dump(classer, open(classer_fn, mode='wb'))
            return classer

    def main(self):
        self.get_sdc_all_tracks()
        self.cluster()


if __name__ == "__main__":
    logger = get_logger()
    ShiftedDeltaClusterer().main()
