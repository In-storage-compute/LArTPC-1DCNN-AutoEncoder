import os
from os import listdir
from os.path import isfile, join
import numpy as np

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

# dataset_x -> noisy signal waveforms
# dataset_y -> clean signal waveforms
# splits data set into subsets based on ADC ranges
def _adc_grouping_helper(dataset_x, dataset_y):

    adc_5_7_x = []
    adc_5_7_y = []

    adc_8_10_x = []
    adc_8_10_y = []

    adc_11_13_x = []
    adc_11_13_y = []

    adc_14_16_x = []
    adc_14_16_y = []

    adc_17_19_x = []
    adc_17_19_y = []

    adc_20_22_x = []
    adc_20_22_y = []

    adc_gt_22_x = []
    adc_gt_22_y = []

    for i, wave in enumerate(dataset_y):
        max_adc = max(wave)
        if max_adc >=5 and max_adc <= 7:
            adc_5_7_x.append(dataset_x[i])
            adc_5_7_y.append(wave)
        elif max_adc >= 8 and max_adc <= 10:
            adc_8_10_x.append(dataset_x[i])
            adc_8_10_y.append(wave)
        elif max_adc >= 11 and max_adc <=13:
            adc_11_13_x.append(dataset_x[i])
            adc_11_13_y.append(wave)
        elif max_adc >= 14 and max_adc <= 16:
            adc_14_16_x.append(dataset_x[i])
            adc_14_16_y.append(wave)
        elif max_adc >= 17 and max_adc <= 19:
            adc_17_19_x.append(dataset_x[i])
            adc_17_19_y.append(wave)
        elif max_adc >= 20 and max_adc <= 22:
            adc_20_22_x.append(dataset_x[i])
            adc_20_22_y.append(wave)
        elif max_adc > 22:
            adc_gt_22_x.append(dataset_x[i])
            adc_gt_22_y.append(wave)

    grouped_waves = [
                    [adc_5_7_x, adc_5_7_y],
                    [adc_8_10_x, adc_8_10_y],
                    [adc_11_13_x, adc_11_13_y],
                    [adc_14_16_x, adc_14_16_y],
                    [adc_17_19_x, adc_17_19_y],
                    [adc_20_22_x, adc_20_22_y],
                    [adc_gt_22_x, adc_gt_22_y]
                    ]
    print(' 0: adc_5_7 \
            1: adc_8_10 \
            2: adc_11_13 \
            3: adc_14_16 \
            4: adc_17_19 \
            5: adc_20_22 \
            6: adc_gt_22 \
          ')
    print()
    

    return grouped_waves

group_num_labels = {
                    0: 'adc_5_7',
                    1: 'adc_8_10',
                    2: 'adc_11_13',
                    3: 'adc_14_16',
                    4: 'adc_17_19',
                    5: 'adc_20_22',
                    6: 'adc_gt_22',
                    }

# prints summary
def adc_grouping(data_x, data_y):
    grouped = _adc_grouping_helper(data_x, data_y)
    sum_ = 0
    
    res = []
    for i in range(7):
        count = len(grouped[i][0])
        print(group_num_labels[i])
        print('{:<12}{}'.format('count', count))
        print()
        res.append(count)
        sum_ += count

    print('{:<15}{}'.format('     Total:', sum_))

    return [grouped, res]