import numpy
from collections import defaultdict
import os
import re
import math
import scipy.io.wavfile as wav
import logging
from collections import Counter
from argparse import ArgumentParser

dates_pattern = re.compile('.*?transcriber_start="(.*?)" transcriber_end="(.*?)".*?')


def extract_ami_intervals(a):
    fh = open(a)
    li = []
    for l in fh:
        matched = dates_pattern.match(l)
        if matched is not None:
            n1 = float(matched.groups()[0])
            n2 = float(matched.groups()[1])
            li.append((n1, n2))
    fh.close()        
    return li        

def get_intervals(a):
    return extract_ami_intervals(a)

def count(all_len, window_size, shift):
    '''
    https://github.com/jameslyons/python_speech_features/
    blob/master/features/sigproc.py#L22
    '''
    return 1 + int(math.ceil((1.0*all_len - window_size)/shift))

def count_frames(wav_file, window_size, shift):
    rate, sig = wav.read(wav_file)
    sig = sig.shape[0]
    all_len = float(sig)/rate
    frame_count = count(all_len, window_size, shift)
    return frame_count

def sec2frame(sec, window_size, shift, frame_count):
    '''
    because of the padding in the end it is somewhat specific to
    //github.com/jameslyons/python_speech_features/
    blob/master/features/sigproc.py 
    '''
    return int(min(math.floor(sec/shift), frame_count))

def get_frames_by_time_interval(pair, window_size, shift, frame_count):
    fr1 = sec2frame(pair[0], window_size, shift, frame_count) 
    fr2 = sec2frame(pair[1], window_size, shift, frame_count)
    return range(fr1, fr2)


def extract_labels(wav_file, annotations, window_size, shift):

    frame_count = count_frames(wav_file, window_size, shift)
    
    frames_annotated = defaultdict(list)
    for i, a in enumerate(annotations):
        print i, a
        intervals = get_intervals(a)
        for pair in intervals:
            frs = get_frames_by_time_interval(
                pair, window_size, shift, frame_count)
            for fr in frs:
                frames_annotated[fr].append(i)
    annotation_array = map_annotations_to_integers(
        frame_count, frames_annotated)
    return annotation_array

def map_annotations_to_integers(frame_count, frames_annotated):
    annotation_list = []
    numpy.zeros(frame_count)
    together_counts = defaultdict(int)
    for fr_i in range(frame_count):
        if len(frames_annotated[fr_i]) == 1:
            annotation_list.append(frames_annotated[fr_i][0] + 1)
        elif len(frames_annotated[fr_i]) > 1:
            #multiple speakers: class 5
            annotation_list.append(5)
            list_shifted = [i+1 for i in frames_annotated[fr_i]]
            together_counts[repr(list_shifted)] += 1
        else:
            # silence
            annotation_list.append(0)
    logging.info('frames with multiple annotations (assignment:5): {}'\
                 .format(together_counts))
    annotation_array = numpy.array(annotation_list)
    logging.info('frames with unique label: {}'.format(Counter(
        annotation_array)))
    return annotation_array        
                

def get_args():
    parser = ArgumentParser()
    parser.add_argument('wav_file')
    parser.add_argument('out_fn')
    parser.add_argument('-w', '--window_size', type=float, default=0.025)
    parser.add_argument('-s', '--shift', type=float, default=0.01)
    parser.add_argument('-a', '--annotations_dir')
    return parser.parse_args()


def main():
    
    import cPickle
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    window_size = args.window_size
    shift = args.shift
    wav_file = args.wav_file
    annotation_dir = args.annotations_dir
    annotation_base_fns = os.listdir(annotation_dir)
    annotations = sorted(['{}/{}'.format(annotation_dir, annotation_base_fn)
    for annotation_base_fn in annotation_base_fns])
    labels = extract_labels(wav_file, annotations, window_size, shift)

    out_fn = args.out_fn
    fh = open(out_fn, 'w')
    cPickle.dump(labels, fh)


if __name__ == "__main__":
    main()
