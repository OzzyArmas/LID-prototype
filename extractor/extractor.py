import python_speech_features
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

TOTAL_FRAMES = 75 #75 ms snippet length
MIN_ENERGY = 12

def get_features(input_file):
    '''
    :param input_file: input wave file to convert to to features
    :returns: an np.array vector (3,__,13) of MFCC features representing the input_file
    '''
    # check for .wav files
    if not input_file[-4:] == '.wav':
        return None
    
    # read .wav file and convert to mfcc
    (rate, sig) = wav.read(input_file)
    
    # mfcc returns frames x 14 matrix
    # where the first column is energy and the remaining 13
    # are mfcc
    features = mfcc(sig, rate)

    features = get_clean_deltas(features)
    return features

def clean_up(features):
    '''
    :param features: mfcc features to clean up
    :returns: clean mfcc features
        function takes mfcc features with values of insufficient energy zero-ed
        then it takes all frames from the first frame that isn't all zero-ed til
        it has as many frames as TOTAL_FRAMES
    '''
    count_frames = 0
    start_frame = -1
    
    for frame,feature in enumerate(features):
        if not feature.all():
            continue
        
        else:
            if start_frame == -1:
                start_frame = frame
            count_frames += 1
        
        if count_frames == TOTAL_FRAMES:
            break
    
    if start_frame > -1:
        return features[start_frame:start_frame + TOTAL_FRAMES]
    return None

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
        if e < MIN_ENERGY:
           features[i] = np.zeros(np.shape(features[i]))
     
    features = clean_up(features)
    
    # check that features are at least as long
    # as the length required
    if features is None or len(features) < TOTAL_FRAMES:
        return None
    
    else:
        # get delta
        delt = delta(features,2)
        # get double delta
        deltdelt = delta(delt,2)
        # return TOTAL_FRAMES x 3 x n_coeff (13)
        return np.concatenate((
                [features],
                [delt], 
                [deltdelt]), 
                axis=0)#note enerhy is a separate output      

def make_feature_set(file_list):
    '''
    :param file_list: list of .wav files to be converted into vectors
    :returns: a list of dimensions 
        (len(file_list) - len(rejected)) x 3 x TOTAL_FRAMES x 13 
    '''   
    feature_set = []
    rejected = []
    
    for idx, input_file in enumerate(file_list):
        feat = get_features(input_file)
        
        if feat is not None:
            feat = np.swapaxes(feat, 0, 1)
            feature_set.append(feat.tolist())
        else:
            rejected.append(idx)
    
    return feature_set, rejected

