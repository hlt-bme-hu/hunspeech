import re
from argparse import ArgumentParser


def speaker_line(t_beg, tdur, name):
    return 'SPEAKER SpeechNonSpeech 1 {} {} <NA> <NA> {} <NA> <NA>'\
            .format(t_beg, tdur, name)


def spkr_info_line(name):
    return 'SPKR-INFO SpeechNonSpeech 1 <NA> <NA> <NA> unknown {} <NA> <NA>'\
            .format(name)


def convert(input_fn, output_fn):
    time_pattern = re.compile(
            '.*?xmin\s=\s([^\n]*).*?xmax\s=\s([^\n]*)' +
                    '.*?text\s=\s"([^\n]*)"', re.DOTALL)
    textgrid = open(input_fn).read()
    output_fh = open(output_fn, 'w')
    intervals = textgrid.split('item [1]:')[1]\
            .split('intervals [')[1:]

    speakers = set([])

    for i in intervals:
        t_beg_, t_end_, name = time_pattern.match(i).groups()
        if len(name) == 0:
            continue
        if name not in speakers:
            output_fh.write('{}\n'.format(spkr_info_line(name)))
            speakers.add(name)
        t_beg = round(float(t_beg_), 3)
        t_end = round(float(t_end_), 3)
        tdur = t_end - t_beg
        output_fh.write('{}\n'.format(speaker_line(t_beg, tdur, name)))


def get_args():

    p = ArgumentParser()
    p.add_argument('input_fn',
            help='TextGrid file with a singe IntervalTier' +
                    ' with annotation for speaker, nonspeech segments' +
                    ' are assumed to be annotated by the empy string')
    p.add_argument('output_fn',
            help='output file in RTTM (NIST Rich Transcription Time ' +
                    'Marked) format, with SPKR-INFO and SPEAKER object types')
    args = p.parse_args()

    return args


def main():

    args = get_args()
    input_fn = args.input_fn
    output_fn = args.output_fn
    convert(input_fn, output_fn)

if __name__ == "__main__":
    main()
