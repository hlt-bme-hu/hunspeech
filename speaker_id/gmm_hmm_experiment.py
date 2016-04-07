from hmmlearn import hmm
import cPickle
from itertools import permutations
import numpy
from sklearn.mixture import GMM
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from scipy.stats import describe
from argparse import ArgumentParser
import sys
import logging

def train_class_gmms(feats, labels, annotation, mixtures):
    data_ordered = map(lambda label:
                       [feats[i] for i in range(len(feats))
                        if annotation[i] == label], labels)
    gmms = []
    for i, d in enumerate(data_ordered):
        logging.info('Training model for class {}'.format(i))
        gmm = GMM(n_components=mixtures)
        gmm.fit(numpy.array(d))
        gmms.append(gmm)
    return gmms    


def calculate_transmat(annotation, length):
    transmat_dict = defaultdict(lambda: defaultdict(int))
    prev = ''
    for actual in list(annotation):
        if prev != '':
            transmat_dict[prev][actual] += 1
        prev = actual
    transmat_array = numpy.array(
        [[float(transmat_dict[i][j]) for j in range(length)]
         for i in range(length)])                            
    transmat_array = normalize(transmat_array, axis=1, norm='l1')
    return transmat_array

def train_empty_model(feat, length, mixtures):
    model = hmm.GMMHMM(n_components=length, n_mix=mixtures, n_iter=1)
    model.fit(feat)
    return model

def supervised_train(feats, annotation,
                     mixtures, dumpdir, model_fn, feats_fn, train_indeces):

    labels = sorted(set(annotation))
    if labels != range(len(labels)):
        sys.stderr.write('not all labels are present')
        quit()
    gmms = train_class_gmms(feats,labels, annotation, mixtures)    
    transmat = calculate_transmat(annotation, len(labels))
    gmm_hmm = train_empty_model(feats[:50], len(labels), mixtures)
    gmm_hmm.gmms_ = gmms
    gmm_hmm.transmat_ = transmat
    if model_fn == None:
        model_fn = '{}/{}_{}_supervised_{}'.format(
            dumpdir, feats_fn, train_indeces, mixtures)
    model_fh = open(model_fn, 'w')    
    cPickle.dump(gmm_hmm, model_fh)    

def unsupervised_train(feats, states, mixtures, iterations, dumpdir,
                        model_fn, feats_fn, train_indeces):
    logging.info('Training GMM-HMM model')
    gmm_hmm = hmm.GMMHMM(n_components=states, n_mix=mixtures,
                         n_iter=iterations)
    gmm_hmm.fit(feats)
    if model_fn == None:
        model_fn = '{}/{}_{}_unsupervised_{}'.format(
            dumpdir, feats_fn, train_indeces, mixtures)
    model_fh = open(model_fn, 'w')
    cPickle.dump(gmm_hmm, model_fh)


def supervised_test(feats, annotation, dumpdir, model):

    model = cPickle.load(open('{}/{}'.format(dumpdir, model)))
    logging.info('Predicting...')
    predicted = model.predict(feats)
    logging.info('Accuracy: {}'.format(accuracy_score(predicted, annotation)))
    logging.info('Confusion matrix: {}'.format(
        confusion_matrix(predicted, annotation)))

def generate_permutations(predicted, num_labels):
    for p in permutations(range(num_labels)):
        dict_ = {}
        for i in p:
            dict_[i] = p[i]
        yield([dict_[j] for j in predicted ])

def unsupervised_test(feats, annotation, dumpdir, model):

    model = cPickle.load(open('{}/{}'.format(dumpdir, model)))
    logging.info('Predicting...')
    predicted = model.predict(feats)
    num_labels = len(set(predicted))
    accuracies = []
    permutations = []
    logging.info('Counting all permutations and calculating accuracies...')
    for p in generate_permutations(predicted, num_labels):
        accuracies.append(accuracy_score(p, annotation))
        permutations.append(p)
    logging.info('Results: {}'.format(describe(accuracies)))
    best_p = permutations[numpy.argmax(accuracies)]
    logging.info('Confusion matrix of best permutation: {}'.format(
        confusion_matrix(best_p, annotation)))

def get_args():
    
    parser = ArgumentParser()
    parser.add_argument('mode',
                        help='train_sup/train_unsup/test_sup/test_unsup')
    parser.add_argument('-f', '--mfcc_feats',
                        help='needed for training as well' +
                        'as for testing')
    parser.add_argument('-m', '--model', help='compulsory for testing')
    parser.add_argument('-a', '--annotation',
                        help='needed for supervised training as well' +
                        'as for testing')
    parser.add_argument('-d', '--dumpdir', default='models',
                        help='here are gmm, gmm-hmm, transmat models saved')
    parser.add_argument('-tr', '--train_indeces', default='0,100000',
                        help='frame indeces of mfcc-feats, separated by ;')
    parser.add_argument('-ts', '--test_indeces', default='100000,0',
                        help='frame indeces of mfcc-feats, separated by ;')
    parser.add_argument('-i', '--iterations', default=10,  help='EM iterations',
                       type=int)
    parser.add_argument('-x', '--mixtures', default=16,  help='GMM mixtures',
                       type=int)
    parser.add_argument('-s', '--states', default=6,
                        help='HMM states, needed for unsupervised training')
    return parser.parse_args()

def main():

    args = get_args()
    feats_fn = args.mfcc_feats
    feats = cPickle.load(open(feats_fn))
    train_start, train_end = args.train_indeces.split(',')
    test_start, test_end = args.test_indeces.split(',')
    train_start = int(train_start)
    if train_end == '':
        train_end = None
    else:    
        train_end = int(train_end)
    test_start = int(test_start)
    if test_end == '':
        test_end = None
    else:
        test_end = int(test_end)
    FORMAT='%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    if args.annotation:
        annotation = cPickle.load(open(args.annotation))
    if args.mode == 'train_sup':
        supervised_train(feats[train_start:train_end],
                        annotation[train_start:train_end], 
                        args.mixtures, args.dumpdir, args.model, 
                        feats_fn, args.train_indeces)
    elif args.mode == 'train_unsup':
        unsupervised_train(feats[train_start:train_end], args.states, 
                          args.mixtures, args.iterations, args.dumpdir,
                          args.model, feats_fn, args.train_indeces)
    elif args.mode == 'test_sup':
        supervised_test(feats[test_start:test_end],
                        annotation[test_start:test_end],
                        args.dumpdir, args.model)
    elif args.mode == 'test_unsup':
        unsupervised_test(feats[test_start:test_end],
                        annotation[test_start:test_end],
                        args.dumpdir, args.model)

if __name__ == "__main__":
    main()
