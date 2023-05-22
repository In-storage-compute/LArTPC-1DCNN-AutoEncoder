import os
from os import listdir
from os.path import isfile, join
import numpy as np


group_num_labels_ar39 = {
                    0: 'adc_4_6',
                    1: 'adc_7_9',
                    2: 'adc_10_12',
                    3: 'adc_13_15',
                    4: 'adc_16_18',
                    5: 'adc_19_21',
                    6: 'adc_gt_21'
                    }

# takes full raw data and extracts waveform of length nticks 
def extract_wave(data, nticks):
    string = 'tck_'
    waveforms = []
    #Here I extract a column in each iteration and append to list
    for i in range(nticks):
        waveforms.append(data[string+str(i)].astype(np.int16))
    #convert to numpy ndarray
    waveforms = np.array(waveforms).astype(np.int16)
    #since raws and columns are inverted we need to transpose it
    return np.transpose(waveforms)

# takes full raw data and returns waveform of length nticks 
def get_std_waveforms(data, nticks):
    #Extract and scale waveform data (passthrough rn)
    raw_waveforms = extract_wave(data, nticks)
    #print(raw_waveforms) 
    #scaled_waveforms = waveform_scaler.fit_transform(raw_waveforms)
    return raw_waveforms

# imposes a min ADC value and filters waves
# takes wavefroms as input
def filter_signal_ADC_min(data, clean_data, adc_min):
    data_wf = []
    clean_wf = []
    for i in range(clean_data.shape[0]):
        if max(abs(clean_data[i])) > adc_min:
            data_wf.append(data[i])
            clean_wf.append(clean_data[i])
    data_wf = np.array(data_wf)
    clean_wf = np.array(clean_wf)
    return data_wf, clean_wf

# data loader
def get_data(wireplane, path):

    path_cc = path+'nu_cc/'
    path_es = path+'nu_es/'
    #noise_path = path+'noise/'
    noise_path = '/home/vlian/Workspace/more-noise/'
    print('nu_cc: ', path_cc)
    print('nu_es: ', path_es)
    print('noise: ', noise_path)
    print('')
    print('----------loading----------')
    sig_name = wireplane+"-signal"
    cln_name = wireplane+"-clnsig"
    
    filenames1 = sorted([path_cc+f for f in listdir(path_cc) if (isfile(join(path_cc, f)) and sig_name in f)])
    clean_filenames1 = sorted([path_cc+f for f in listdir(path_cc) if (isfile(join(path_cc, f)) and cln_name in f)])
    filenames2 = sorted([path_es+f for f in listdir(path_es) if (isfile(join(path_es, f)) and sig_name in f)])
    clean_filenames2 = sorted([path_es+f for f in listdir(path_es) if (isfile(join(path_es, f)) and cln_name in f)])
    filenames =  filenames1+filenames2
    clean_filenames = clean_filenames1+clean_filenames2
    noise_filenames = sorted([f for f in listdir(noise_path) if (isfile(join(noise_path, f)) and wireplane in f)])

    combined_data = np.concatenate([np.load(fname, mmap_mode='r') for fname in filenames])
    combined_clean_data = np.concatenate([np.load(fname, mmap_mode='r') for fname in clean_filenames])
    combined_noise = np.concatenate([np.load(noise_path+fname, mmap_mode='r') for fname in noise_filenames], )
    print('--------data loaded!-------')

    return combined_data, combined_clean_data, combined_noise


# dataset_x -> noisy signal waveforms
# dataset_y -> clean signal waveforms
# splits data set into subsets based on ADC ranges
def _adc_grouping_helper(dataset_x, dataset_y):
    adc_4_6_x = []
    adc_4_6_y = []

    adc_7_9_x = []
    adc_7_9_y = []

    adc_10_12_x = []
    adc_10_12_y = []

    adc_13_15_x = []
    adc_13_15_y = []

    adc_16_18_x = []
    adc_16_18_y = []

    adc_19_21_x = []
    adc_19_21_y = []

    adc_gt_21_x = []
    adc_gt_21_y = []

    noise_x = []
    noise_y = []

    for i, wave in enumerate(dataset_y):
        if sum(abs(wave)) == 0:
            noise_x.append(dataset_x[i])
            noise_y.append(wave)
            continue
        max_adc = max(abs(wave))
        if max_adc >= 4 and max_adc <= 6:
            adc_4_6_x.append(dataset_x[i])
            adc_4_6_y.append(wave)
        elif max_adc >= 7 and max_adc <=9:
            adc_7_9_x.append(dataset_x[i])
            adc_7_9_y.append(wave)
        elif max_adc >= 10 and max_adc <= 12:
            adc_10_12_x.append(dataset_x[i])
            adc_10_12_y.append(wave)
        elif max_adc >= 13 and max_adc <= 15:
            adc_13_15_x.append(dataset_x[i])
            adc_13_15_y.append(wave)
        elif max_adc >= 16 and max_adc <= 18:
            adc_16_18_x.append(dataset_x[i])
            adc_16_18_y.append(wave)
        elif max_adc >= 19 and max_adc <= 21:
            adc_19_21_x.append(dataset_x[i])
            adc_19_21_y.append(wave)
        else:
            adc_gt_21_x.append(dataset_x[i])
            adc_gt_21_y.append(wave)

    grouped_waves = [
                    [adc_4_6_x, adc_4_6_y],
                    [adc_7_9_x, adc_7_9_y],
                    [adc_10_12_x, adc_10_12_y],
                    [adc_13_15_x, adc_13_15_y],
                    [adc_16_18_x, adc_16_18_y],
                    [adc_19_21_x, adc_19_21_y],
                    [adc_gt_21_x, adc_gt_21_y],
                    [noise_x, noise_y]  
                    ]
    print(' 0: adc_4_6 \
            1: adc_7_9 \
            2: adc_10_12 \
            3: adc_13_15 \
            4: adc_16_18 \
            5: adc_19_21 \
            6: adc_gt_21\
            7: noise')
    print()
    

    return grouped_waves

group_num_labels = {
                    0: 'adc_4_6',
                    1: 'adc_6_9',
                    2: 'adc_10_12',
                    3: 'adc_13_15',
                    4: 'adc_16_18',
                    5: 'adc_19_21',
                    6: 'adc_gt_21',
                    7: 'noise'
                    }

# prints summary
def adc_grouping(data_x, data_y):
    grouped = _adc_grouping_helper(data_x, data_y)
    sum_ = 0
    
    res = []
    for i in range(8):
        count = len(grouped[i][0])
        print(group_num_labels[i])
        print('{:<12}{}'.format('count', count))
        print()
        res.append(count)
        if i < 7:
            sum_ += count

    print('{:<15}{}'.format('     Total:', sum_))

    return [grouped, res]

def process_data(wireplane,path,ADC_MIN):
    nticks = 200
    # load raw data
    combined_data, combined_clean_data, combined_noise = get_data(wireplane, path)
    print('---------------------------------')
    print('     signal+noise: ', len(combined_data))
    print('     clean signal: ', len(combined_clean_data))
    print('     noise       : ', len(combined_noise))
    print('---------------------------------')

    # extract waveforms
    signal_waveforms = get_std_waveforms(combined_data, nticks)
    clean_signal_waveforms = get_std_waveforms(combined_clean_data, nticks)  # for autoencoder
    print('')
    print('filtering out small signals --> ADC >', ADC_MIN)
    print('     noise+signal : ', signal_waveforms.shape)
    print('     clean signal : ', clean_signal_waveforms.shape)
    #Filter out tiny signals < ADC_MIN, but leave big signals to test on (incl > ADC_MAX)
    signal_waveforms, clean_signal_waveforms = filter_signal_ADC_min(signal_waveforms,
                                                clean_signal_waveforms, ADC_MIN)
    print('------------after filtering------------')
    print('     noise+signal : ', signal_waveforms.shape)
    print('     clean signal : ', clean_signal_waveforms.shape)

    noise_waveforms = get_std_waveforms(combined_noise, nticks)
    noiseless_waveform = noise_waveforms*0 # for autoencoder

    return signal_waveforms, clean_signal_waveforms, noise_waveforms, noiseless_waveform