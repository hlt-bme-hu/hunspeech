# emSpeech

emSpeech is the speech module of the [e-magyar](http://e-magyar.hu/) language processing system.

## Speech activity detection (emSad)

The speech activity detection module wraps the [SHOUT](http://shout-toolkit.sourceforge.net/).

## Speaker diarization (emDia)

The speaker diarization module wraps the 'shout_segment' and 'shout_cluster' programs of
[SHOUT](http://shout-toolkit.sourceforge.net/), with added audio conversion using the
SoX (Sound eXchange) utility.
For details on the usage run
~~~
python speaker_diarization/em-dia.py --help
~~~

## Installation

1. Install SHOUT. Instructions can be found [here](http://shout-toolkit.sourceforge.net/download.html).
1. Install sox. sox should be available on most Linux distribution (e.g. `apt-get install sox`).
1. Check out this repository.
1. Set the `SHOUT_DIR` environmental variable to the directory where you isntalled SHOUT:

    export SHOUT_DIR=/path/to/shout

This will be the default place to look for the model files SHOUT needs.

## Citation

Please cite [Marijn Huijbregts's dissertation](http://shout-toolkit.sourceforge.net/img/thesis_Marijn_Huijbregts.pdf) on SHOUT.

