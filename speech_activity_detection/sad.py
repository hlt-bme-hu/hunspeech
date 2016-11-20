#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the GPL license.

from argparse import ArgumentParser
import os
import subprocess


class EMSpeechActicityDetection:
    """Speech activity detection and segmentation

    This class is a wrapper for the SHOUT toolkit's SAD module.
    Since SHOUT expects the input to be raw audio, it is first converted
    into the correct raw format by sox (Sound eXchange),
    then shout_segment is called.
    SHOUT outputs a single segmentation file, which is saved to
    segments.txt by default.
    Each segment is labeled as SPEECH, SIL (silence) or SOUND.
    EMSpeechActicityDetection supports two additional saving solutions:
        1. segment the input according to SHOUT's segmentation into
        individual audio files.
        2. group segments by labels and concatenate them into a single file.
        This produces at most three files: one containing all speech, one
        containing all silence and another one containing all sound.
    """

    def __init__(self, filename, model=None, segment_out='segments.txt',
                 segment_dir=None, shout_path=os.environ.get('SHOUT_DIR')):
        self.filename = filename
        if model is None:
            self.model = os.path.join(os.environ.get('SHOUT_DIR'),
                                      'models', 'shout.sad')
        else:
            self.model = model
        self.segment_out = segment_out
        self.binary_path = '{}/shout_segment'.format(shout_path)

    def segment(self):
        self.raw_filename = EMSpeechActicityDetection.convert_to_raw(
            self.filename)
        cmd = '{0} -a {1} --am-segment {2} -mo {3}'.format(
            self.binary_path,
            self.raw_filename,
            self.model,
            self.segment_out,
        )
        subprocess.call(cmd, shell=True)

    @staticmethod
    def convert_to_raw(filename):
        """ accepts mp3, wav and raw files """
        EMSpeechActicityDetection.__check_audio_file(filename)
        basename, ext = os.path.splitext(filename)
        if ext == '.mp3':
            EMSpeechActicityDetection.convert_mp3_to_wav(filename)
        fn = EMSpeechActicityDetection.convert_wav_to_raw('{0}.wav'.format(
            basename))
        return fn

    @staticmethod
    def convert_mp3_to_wav(filename):
        basename, ext = os.path.splitext(filename)
        out_fn = '{}.wav'.format(basename)
        subprocess.call('sox {0} {1}'.format(filename, out_fn), shell=True)
        return out_fn

    @staticmethod
    def convert_wav_to_raw(filename):
        basename, ext = os.path.splitext(filename)
        out_fn = '{}.raw'.format(basename)
        params = '-r 16k -b 16 -L -c 1'
        subprocess.call('sox {0} {1} {2}'.format(params, filename, out_fn),
                        shell=True)
        return out_fn

    @staticmethod
    def __check_audio_file(filename):
        if not os.path.exists(filename):
            raise Exception('Source file does not exist: {}'.format(
                filename))
        ext = os.path.splitext(filename)[-1]
        if ext not in ('.raw', '.mp3', '.wav'):
            raise ValueError('Cannot handle [{0}] files'.format(
                ext))


def parse_args():
    p = ArgumentParser()
    p.add_argument('-i', '--input', type=str,
                   help='Input file. Use this option if you want to segment'
                   ' a single file'
                   )
    p.add_argument('-m', '--model', type=str,
                   help='SHOUT acoustic model',
                   default='{}/shout_am.sad'.format(
                       os.environ.get('SHOUT_DIR')),
                   )
    p.add_argument('-o', '--segment-out', type=str,
                   help='Write segments to file',
                   default='segments.txt'
                   )
    return p.parse_args()


def main():
    args = parse_args()
    sad = EMSpeechActicityDetection(filename=args.input, model=args.model,
                                    segment_out=args.segment_out)
    sad.segment()

if __name__ == '__main__':
    main()
