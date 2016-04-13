from features.base import mfcc
import scipy.io.wavfile as wav
from argparse import ArgumentParser
import cPickle

def get_args():
    
    parser = ArgumentParser()
    parser.add_argument('wav_file')
    parser.add_argument('out_fn')
    parser.add_argument('-w', '--window_size', type=float, default=0.025)
    parser.add_argument('-s', '--shift', type=float, default=0.01)
    return parser.parse_args()


def main():

    args = get_args()
    (_,sig) = wav.read(args.wav_file)
    features = mfcc(sig, winlen=args.window_size, winstep=args.shift)
    fh = open(args.out_fn, 'w')
    cPickle.dump(features, fh)


main()    
