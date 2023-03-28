import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf

def load_data(path, wireplane):
    x_train = np.load(path+'x_train_V.npy')
    x_test = np.load(path+'x_test_V.npy')
    y_train = np.load(path+'y_train_AE_V.npy')
    y_test = np.load(path+'y_test_AE_V.npy')
    mean = x_train.mean()
    std = x_train.std()

    x_train_scaled = (x_train-mean)/std
    x_test_scaled = (x_test-mean)/std

    y_train_scaled = []
    for i in range(len(y_train)):
        if sum(y_train[i]) == 0:
            y_train_scaled.append(y_train[i])
        else:
            scaled = (y_train[i]-mean)/std
            y_train_scaled.append(scaled)
    y_train_scaled = np.array(y_train_scaled)

    y_test_scaled = []
    for i in range(len(y_test)):
        if sum(y_test[i]) == 0:
            y_test_scaled.append(y_test[i])
        else:
            scaled = (y_test[i]-mean)/std
            y_test_scaled.append(scaled)
    y_test_scaled = np.array(y_test_scaled)

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, mean, std

def combine_overlapping_ranges(ranges):
    # sort the original list in ascending order according to the first element in each sublist
    ranges = sorted(ranges, key=lambda x: x[0])

    # initialize the result list
    result = []

    # iterate over the sorted list and combine overlapping sublists
    for range in ranges:
        if result and range[0] <= result[-1][1]:
            # combine the current sublist with the previous one
            result[-1] = [min(range[0], result[-1][0]), max(range[1], result[-1][1])]
        else:
            # no overlap, so simply append the current sublist to the result
            result.append(range)

    return result

# takes single wave (array), and baseline (the value at which the wave is flat) and returns
# a list of ranges where there are signals waveforms
from scipy.signal import find_peaks
def find_peak_range(array):
    baseline = 0.001  # after scaling and rescaling, bl not at 0 exactly
    array = abs(array)
    # find the peaks
    peaks, _ = find_peaks(array)
    peak_ranges = []
    start_spc = 0
    end_spc = 0

    if abs(array[0]) > baseline:
        end_spc = 0
        while end_spc < 200 and ((array[end_spc] > baseline) or (array[end_spc] < baseline)):
            end_spc += 1

    for peak in peaks:
        if peak == end_spc:
            end = peak
            while end < 200 and array[end] > abs(baseline):
                end += 1
            if [start_spc, end] not in peak_ranges:
                peak_ranges.append([start_spc, end])

        else:
            start = peak
            while start > 0 and array[start] > abs(baseline):
                start -= 1
        
            end = peak
            while end < 200 and array[end] > abs(baseline):
                end += 1
            if [start, end] not in peak_ranges:
                peak_ranges.append([start, end])

    peak_ranges = combine_overlapping_ranges(peak_ranges)

    return peak_ranges

def merge_ranges(wave, bound):
    # range_b1 is the ranges after merging using bound = 1
    range_b1 = find_peak_range(wave)
    idx = 0
    while idx < len(range_b1) - 1:
        if range_b1[idx][1] + bound >= range_b1[idx+1][0]:
            range_b1[idx][1] = max(range_b1[idx][1], range_b1[idx+1][1])
            del range_b1[idx+1]
        else:
            idx += 1
    return range_b1


# find ranges where there are no signals using ranges
# where there are signals as input
def get_non_signal_ranges(signal_ranges):


    output = []
    
    for single_wave_ranges in signal_ranges:
        single_out = []
        start = 0
        # signal_ranges.sort()  # more efficient for unsorted input
        for subrange in single_wave_ranges:
            if start < subrange[0]:
                single_out.append([start, subrange[0]])
            start = subrange[1]
            # start = max(subrange[0], start)  # more efficient for unsorted input
        if start < 200:
            single_out.append([start, 200])
        output.append(single_out)
    return output
    

# helper to calculate mse of segments or full wave
def calculate_single_mse_helper(seqment_wave, seqment_pred_wave):
    #single_mse = np.mean((seqment_wave-seqment_pred_wave)**2)
    single_mse = tf.math.reduce_mean(tf.math.square(seqment_wave - seqment_pred_wave)) 
    #single_mse = np.float32(single_mse) 
    return single_mse

# helper method to calculate mse given some ranges
def segment_mse_helper(expected, prediction, signal_ranges, non_sig_ranges):
    sig_mse = 0
    no_sig_mse = 0
    for range_ in signal_ranges:
        expected_wave = expected[range_[0]:range_[1]]
        pred_wave = prediction[range_[0]:range_[1]]
        sig_mse += calculate_single_mse_helper(expected_wave, pred_wave)
    
    for range_ in non_sig_ranges:
        expected_wave = expected[range_[0]:range_[1]]
        pred_wave = prediction[range_[0]:range_[1]]
        no_sig_mse += calculate_single_mse_helper(expected_wave, pred_wave)
    
    w_1, w_2 = 1, 0.7

    mse = w_1*sig_mse + w_2*no_sig_mse

    return mse

def calculate_single_mse(expected, prediction, signal_ranges, non_sig_ranges):
    
    signal_region = segment_mse_helper(expected, prediction, signal_ranges, non_sig_ranges)

    return signal_region

# -----------------------------------------------------------------------------------------------------