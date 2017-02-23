# Tensorflow CTC on LibriSpeech (Not fully functional, still working :-) )

An example trying to run baidu asr system "https://github.com/baidu-research/ba-dls-deepspeech" using the default ctc function tensorflow. 
Similar to the Baidu system I am using LibriSpeech dataset.
The project is not complete. 

As of Feb 22, 2017 
 * The GPU I am using are running out of memory. ( tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator) Pool allocator is taking a lot of time if I train it on more than few 100's of audio files.
 * I am still testing, so the Neural Network is not deep. I am using only train dataset and no test or validation data sets. I am trying to get the network to memorize train dataset.

## Data
I made use of the LibriSpeech ASR corpus. 
How to download dataset and create json files is described in the above mentioned baidu github. 
I used the audio files of speaker ID 19 to train my system.

## File Organization
  ctc_tensorflow_example.py, ctc_tensorflow_multidata_example.py, utils.py are from the forked directory
  ctc_spectrogram_single.py downlaods one timit dataset wavefile and uses its spectrogram to train NN
  ctc_spectrogram_multidata.py takes multiple audio files as input (train_corpus.json) 
  train_corpus.json > from baidu asr


## Requirements

- Python 2.7+
- Tensorflow 0.12
- python_speech_features (not necessary if using spectrogram)
- numpy
- scipy

## License

This project is licensed under the terms of the MIT license.
