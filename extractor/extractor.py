from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

TOTAL_FRAMES = 50 # of frames per chunk
MIN_ENERGY = 12
MAX_EMERGY = 11 # to be used for noise data

def get_noise(input_file, chunk = False):
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
    # where the first column is energy
    # and the remaining 13 are mfcc
    features = mfcc(sig, rate)
    features = remove_speech(features)
    if chunk:
        out_feats = []
        for feat in range(TOTAL_FRAMES, len(features), TOTAL_FRAMES):
            out_feats.append(get_deltas(features[features[last_feat:feat]]))
        features = out_feats
    return features

def remove_speech(features):
    energy = features[:,0]
    features = features[:,1:]

    # Zeros non-speech vectors
    for i,e in enumerate(energy):
        if e > MAX_ENERGY:
           features[i] = np.zeros(np.shape(features[i]))

    return features


def get_features(input_file, chunk):
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
    # where the first column is energy
    # and the remaining 13 are mfcc
    features = mfcc(sig, rate)
    features = filter_energy(features)
    if len(features) > 0:
        if chunk:
            out_feats = []
            last_feat = 0
            for feat in range(TOTAL_FRAMES, len(features), TOTAL_FRAMES):
                out_feats.append(get_deltas(features[last_feat:feat]))
                last_feat = feat
            # at this point out_feats is either []
            # or it's n_subsequences x 3 x TOTAL_FRAMES x 13
            return out_feats
        else:
            return get_deltas(features)
    
    return []

def filter_energy(features):
    '''
    :param features: np.array of mfcc features
    :return: np.array of mfcc features after` energy is filtered
    '''
    energy = features[:,0]
    features = features[:,1:]

    # Zeros non-speech vectors
    for i,e in enumerate(energy):
        if e < MIN_ENERGY:
           features[i] = np.zeros(np.shape(features[i]))
     
    # Cuts initial and tail noise 
    features = clean_up(features)
    # clean up reverse file, then return it to oringal order
    return clean_up(features[::-1])[::-1] if len(features) > 0 else []

   
def clean_up(features):
    '''
    :param features: mfcc features to clean up
    :returns: clean mfcc features
        function takes mfcc features with values of insufficient energy zero-ed
        then it takes all frames from the first frame that isn't all 
        zero-ed until it has as many frames as TOTAL_FRAMES
    '''
    count_frames = 0
    start_frame = -1

    for frame,feature in enumerate(features):
        if feature.all():
            if start_frame == -1:
                start_frame = frame
            count_frames += 1
    
    if count_frames >= TOTAL_FRAMES:
        features = features[start_frame:]
        return features
    
    return []

def get_deltas(features):
    '''
    :param features: MFCC feature array to get a snippet from
    :return: array of dimensions 3 x TOTAL_FRAMES x 13  
        representing ceptral features for length ms of time or 
        None if the min length can't be met
    '''
    # get delta
    delt = delta(features,2)
    # get double delta
    deltdelt = delta(delt,2)
    # return  3 x TOTAL_FRAMES x 13
    return np.concatenate(([features], [delt], [deltdelt]), axis = 0)


def make_feature_set(file_list, language_label, chunk = True):
    '''
    :param file_list: list of .wav files to be converted into vectors
    :returns: a list of dimensions 
        (len(file_list) - len(rejected files)) x 3 x TOTAL_FRAMES x 13 
    '''
    feature_set = []
    for idx, input_file in enumerate(file_list):
        # at this point out_feats is either []
        # or it's n_subsequences x 3 x TOTAL_FRAMES x 13
        feat = get_features(input_file, chunk)
        if len(feat) > 0:
            feature_set.append(feat)

    # Now, feature_set will be
    # n_files x variable n_subsequences x 3 x TOTAL_FRAMES x 13
    # which is why we concatenate along axis = 0 to get
    # sum(sub_sequences) x 3 x TOTAL_FRAMES x 13
    if chunk:
        feature_set = np.concatenate(feature_set, axis=0)

    # Given that files are all the same language per directory
    # iterating through a directory and creating all subsequences will
    # result on sum(sub_sequences) of langugage_labels
    label_vector = [language_label] * len(feature_set)
    
    return feature_set, label_vector

def get_noise_set(file_list, noise_label, chunk = True):
    '''
    Very similar method to the one above, except it gets
        noise instead of speech
    :param file_list: list of files to parse
    :param noise_label: label to indicate noise
    :param chunk: whether or not to separate noise data into chunks
    :return: (noise_set, noise_label), tuple of lists
    '''
    noise_set = []
    for idx, input_file in enumerate(file_list):
        feat = get_noise(input_file, chunk)

    if chunk:
        noise_set  = np.concatenate(noise_set, axis = 1)
    
    noise_label = [noise_label] * len(noise_set)

    return noise_set, noise_label