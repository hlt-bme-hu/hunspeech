import wave
from argparse import ArgumentParser


def parse_args():
    p = ArgumentParser()
    p.add_argument('wav', help='wave file to segment', type=str)
    p.add_argument('seg',
                   help='speech activity detection/diarization output created by SHOUT',
                   type=str)
    p.add_argument('-s', '--segments-out', help='output directory for '
                   'segments.  Each segment will appear in a single file',
                   type=str, dest='segdir',
                   default='segments')
    p.add_argument('-j', '--segments-joined', help='different types of segments ('
                   'speech, sil, sound for SAD, those defined by different speakers '
                  ' for diarization) are concatenated and written to'
                   'separate files to this directory',
                   type=str, dest='joindir',
                   default='segments_joined')
    return p.parse_args()


def read_segments(fn):
    segments = []
    with open(fn) as f:
        for l in f:
            fd = l.strip().split(' ')
            if not fd[0] == 'SPEAKER':
                continue
            start = float(fd[3])
            duration = float(fd[4])
            typ = fd[7]
            segments.append((start, duration, typ))
    return segments


def segment_and_save_audio(audio_fn, segments, outdir):
    w = wave.open(audio_fn)
    framerate = w.getframerate()
    prev_end = 0
    for i, (start, dur, typ) in enumerate(segments):
        # for diarization there can be a gap between two interval of interest
        not_needed_frames = int((start - prev_end) * framerate)
        _ = w.readframes(not_needed_frames)
        prev_end = start + dur
        n = (int)(dur * framerate)
        d = w.readframes(n)
        outw = wave.open('{0}/{1}_{2}.wav'.format(outdir, i, typ.lower()), 'w')
        outw.setparams(w.getparams())
        outw.writeframes(d)
        outw.close()
    w.close()


def split_and_join_similar(audio_fn, segments, outdir):
    w = wave.open(audio_fn)
    framerate = w.getframerate()
    outw = {}
    alltyp = set([s[2] for s in segments])
    for typ in alltyp:
        outw[typ.lower()] = wave.open(
            '{0}/{1}.wav'.format(outdir, typ.lower()), 'w')
    for v in outw.values():
        v.setparams(w.getparams())
    prev_end = 0    
    for i, (start, dur, typ) in enumerate(segments):
        not_needed_frames = int((start - prev_end) * framerate)
        _ = w.readframes(not_needed_frames)
        prev_end = start + dur
        n = (int)(dur * framerate)
        d = w.readframes(n)
        outw[typ.lower()].writeframes(d)
    for v in outw.values():
        v.close()
    w.close()


def main():
    args = parse_args()
    seg_fn = args.seg
    audio_fn = args.wav
    seg_dir = args.segdir
    join_dir = args.joindir
    segments = read_segments(seg_fn)
    segment_and_save_audio(audio_fn, segments, seg_dir)
    split_and_join_similar(audio_fn, segments, join_dir)

if __name__ == '__main__':
    main()
