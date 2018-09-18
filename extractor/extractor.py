'''
make sure to download and import files for MFCC
'''
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

time_offset = 24 # start at 25th frame
snippet_length = 75 #75 ms snippet length

def get_features(input_file, offset=time_offset, length=snippet_length):
    '''
    :param input_file: input wave file to convert to to features
    :returns: an np.array vector (3,__,13) of MFCC features representing the input_file
    '''
    if not input_file[-4:] == '.wav':
        return None
    (rate, sig) = wav.read(input_file)
    features = mfcc(sig, rate)
    delt = delta(features, 2)
    deltdelt = delta(delt, 2)

    '''
    Mel Frequency Cepstral Coefficients and it's deltas
    TODO: look into shifted deltas (SDS)
    '''
    
    return clean_samples(features, delt, 
                        deltdelt,
                        offset=time_offset, 
                        length=snippet_length)

def clean_samples(features, delt, deltdelt, offset=time_offset, length=snippet_length):
    '''
    :param features: MFCC feature to get a snippet from
    :param delt: Delta features to get a snippet from
    :param deltdelt: DeltaDelta features to get a snippet from
    :return: 3 x 13 array representing ceptral features for length ms of time or None if
    the length can't be met
    '''
    features = features[offset:offset + length]
    delt = delt[offset:offset + length]
    deltdelt = deltdelt[offset:offset + length]
    return np.concatenate(([features], [delt], [deltdelt]), axis=0) if (len(features) == length) else  None

def make_feature_set(file_list):
    '''
    :param file_list: list of .wav files to be converted into vectors
    :returns:
    '''    
    feature_set = []

    for input_file in file_list:
        feat = get_features(input_file)
        if feat is not None:
            feature_set.append(feat)

    return feature_set

