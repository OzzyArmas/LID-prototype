import importlib
import python_speech_features
importlib.reload(python_speech_features)
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

total_frames = 75 #75 ms snippet length
energy_threshold = 12

def get_features(input_file):
    '''
    :param input_file: input wave file to convert to to features
    :returns: an np.array vector (3,__,13) of MFCC features representing the input_file
    '''
    if not input_file[-4:] == '.wav':
        return None
    (rate, sig) = wav.read(input_file)
    features = mfcc(sig, rate)

    '''
    Mel Frequency Cepstral Coefficients and it's deltas
    TODO: look into shifted deltas (SDC)
    '''
    features = get_clean_deltas(features)
    return features

def clean_up(features):
    '''
    :param features: mfcc features to clean up
    :returns: clean mfcc features
        function takes mfcc features with values of insufficient energy zero-ed
        then it takes all frames from the first frame that isn't all zero-ed til
        it has as many frames as total_frames
    '''
    count_frames = 0
    frames_total = total_frames
    start_frame = None
    for frame,feature in enumerate(features):
        if not feature.all():
            continue
        else:
            if start_frame == None:
                start_frame = frame
            count_frames += 1
        if count_frames == total_frames:
            break
    return features[start_frame:start_frame + frames_total]

def get_clean_deltas(features):
    '''
    :param features: MFCC feature to get a snippet from
    :param delt: Delta features to get a snippet from
    :param deltdelt: DeltaDelta features to get a snippet from
    :return: 3 x max(length, full audio) x 13 array 
        representing ceptral features for length ms of time or 
        None if the min length can't be met
    '''
    energy = features[:,0]
    features = features[:,1:]
            
    # experimental for loop to test if eliminating frames
    # that aren't loud enough can be benficial for classification

    for i,e in enumerate(energy):
        if e < energy_threshold:
           features[i] = np.zeros(np.shape(features[i]))
     
    features = clean_up(features)
    
    delt = delta(features,2)
    deltdelt = delta(delt,2)
    # check that features are at least as long
    # as the length required
    if len(features) >= total_frames:
        return (np.concatenate((
            [features],
            [delt], 
            [deltdelt]), 
            axis=0), energy) #note enerhy is a separate output
    else:
        # return None if file is shorter than snippet
        return (None, None)      

def make_feature_set(file_list):
    '''
    :param file_list: list of .wav files to be converted into vectors
    :returns:
    '''   
    feature_set = []
    for input_file in file_list:
        feat = get_features(input_file)[0]
        if feat is not None:
            feat = np.swapaxes(feat, 0, 1)
            feature_set.append(feat.tolist())
    
    return feature_set

