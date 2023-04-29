"""
Van Tha Bik Lian
Jan. 03, 2023

preprocesses LArTPC waveform data for training
and testing 1dcnn ROI finder and denoising autencoder
based on the 1dcnn

preprocess for 1D CNN & AUTOENCODER

1) Load data
2) Split data 50:50 for (training+val; 80:20) & testing
3) get mean + std and scale datasets
4) Save processed data to be used in training 1D CNN and AutoEncoder
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import numpy as np
from os import listdir
from os.path import isfile, join

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

# imposes a max ADC value and filters waves
# takes wavefroms as input
def filter_signal_ADC_max(data, clean_data, adc_max):
    data_wf = []
    clean_wf = []
    for i in range(clean_data.shape[0]):
        if max(clean_data[i]) < adc_max:
            data_wf.append(data[i])
            clean_wf.append(clean_data[i])
    data_wf = np.array(data_wf)
    clean_wf = np.array(clean_wf)
    return data_wf, clean_wf

# imposes a min ADC value and filters waves
# takes wavefroms as input
def filter_signal_ADC_min(data, clean_data, adc_min):
    data_wf = []
    clean_wf = []
    for i in range(clean_data.shape[0]):
        if max(clean_data[i]) > adc_min:
            data_wf.append(data[i])
            clean_wf.append(clean_data[i])
    data_wf = np.array(data_wf)
    clean_wf = np.array(clean_wf)
    return data_wf, clean_wf


# data loader
def get_data(wireplane):

    path_cc = "C:\\Users\\vanth\\Desktop\\Workspace\\train_dune_lartpc_v2\\nu_cc\\"
    path_es = "C:\\Users\\vanth\\Desktop\\Workspace\\train_dune_lartpc_v2\\nu_es\\"
    noise_path = "C:\\Users\\vanth\\Desktop\\Workspace\\train_dune_lartpc_v2\\noise\\"
    print('nu_cc: ', path_cc)
    print('nu_es: ', path_es)
    print('noise: ', noise_path)
    print('')
    print('----------loading----------')
    sig_name = wireplane+"-signal"
    cln_name = wireplane+"-clnsig"
    
    filenames1 = [path_cc+f for f in listdir(path_cc) if (isfile(join(path_cc, f)) and sig_name in f)]
    clean_filenames1 = [path_cc+f for f in listdir(path_cc) if (isfile(join(path_cc, f)) and cln_name in f)]
    filenames2 = [path_es+f for f in listdir(path_es) if (isfile(join(path_es, f)) and sig_name in f)]
    clean_filenames2 = [path_es+f for f in listdir(path_es) if (isfile(join(path_es, f)) and cln_name in f)]
    filenames =  filenames1+filenames2
    clean_filenames = clean_filenames1+clean_filenames2
    noise_filenames = [f for f in listdir(noise_path) if (isfile(join(noise_path, f)) and wireplane in f)]

    combined_data = np.concatenate([np.load(fname) for fname in filenames])
    combined_clean_data = np.concatenate([np.load(fname) for fname in clean_filenames])
    combined_noise = np.concatenate([np.load(noise_path+fname) for fname in noise_filenames])
    print('--------data loaded!-------')

    return combined_data, combined_clean_data, combined_noise


def main():
    np.random.seed(42)
    # call functions
    args = sys.argv[1:]
    planes_ = ['U', 'V', 'Z']
    if len(args) == 2 and args[0] == '-plane' and args[1] in planes_:
        wireplane = args[1]
        nticks = 200
        ADC_MIN = 3
        print('Plane:', wireplane, 'window size: ', nticks)

        # load raw data
        combined_data, combined_clean_data, combined_noise = get_data(wireplane)
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
        print('---------------------------------------')
        print('waveforms split into training and testing set')
        test_size = 0.5
        print('     test_size: ', test_size)
        #split train and test sets ()
        signal_waveforms, test_signal_waveforms, clean_signal_waveforms, test_clean_waveforms  = train_test_split(
                                                signal_waveforms, clean_signal_waveforms, test_size=test_size, shuffle=True
                                                )
        noise_waveforms = get_std_waveforms(combined_noise, nticks)
        noiseless_waveform = noise_waveforms*0 # for autoencoder
        print('')
        print('--> noise waveforms extracted, targets (array of zeros) generated for AUTOENCODER')

        #generate y data (assuming all radiologicals contain signal, all noise does not)
        y_noise_full_ROI = np.zeros(noise_waveforms.shape[0])  # for 1dcnn
        y_signal_ROI = np.ones(signal_waveforms.shape[0])      # for 1dcnn
        y_test_signal_ROI = np.ones(test_signal_waveforms.shape[0])  # for 1dcnn
        print('--> targets (arrays of ones and zeros) generated for 1DCNN')

        # split test and train noise datasets (50k)
        x_noise_train, x_noise_test, y_noise_train_ROI, y_noise_test_ROI, y_noise_train_AE, y_noise_test_AE = train_test_split(
            noise_waveforms, y_noise_full_ROI, noiseless_waveform, test_size=test_size, shuffle=True
        )
        print('--> noise waveforms split into training and testing set. test size:', test_size)

        #Shuffle signal waveforms to be safe
        signal_waveforms, y_signal_RIO, y_signal_AE = shuffle(signal_waveforms,y_signal_ROI, clean_signal_waveforms)  # train
        x_test, y_test_ROI, y_test_AE = shuffle(test_signal_waveforms, y_test_signal_ROI, test_clean_waveforms)       # test
        print('--> shuffled signal waveforms (both testing & training set)')

        print('')
        print('     x_train            (1DCNN + AE) : ', signal_waveforms.shape)
        print('     x_test             (1DCNN + AE) : ', test_signal_waveforms.shape)
        print('     truth signal train (1DCNN)      : ', y_signal_RIO.shape)
        print('     truth singal test  (1DCNN)      : ', y_test_signal_ROI.shape)
        print('     truth signal train (AE)         : ', clean_signal_waveforms.shape)
        print('     truth signal test  (AE)         : ', test_clean_waveforms.shape)

        print('')
        print('     noise waveforms            : ', noise_waveforms.shape)
        print('     x_noise_train (1DCNN + AE) : ', x_noise_train.shape)
        print('     x_noise_test  (1DCCN + AE) : ', x_noise_test.shape)
        print('     y_noise_train (1DCNN)      : ', y_noise_train_ROI.shape)
        print('     y_noise_test  (1DCNN)      : ', y_noise_test_ROI.shape)
        print('     y_noise_train (AE)         : ', y_noise_train_AE.shape)
        print('     y_noise_test  (AE)         : ', y_noise_test_AE.shape)
        print('')

        print('--> Select First n signal samples from (shuffled) set of signals where n = noise samples for balanced train set')
        #Select First n signal samples from (shuffled) set of signals where n = noise samples for balanced train set 
        x_train = np.concatenate((signal_waveforms[:int(x_noise_train.shape[0])], x_noise_train))
        y_train_ROI = np.concatenate((y_signal_RIO[:int(x_noise_train.shape[0])], y_noise_train_ROI))
        y_train_AE = np.concatenate((y_signal_AE[:int(x_noise_train.shape[0])], y_noise_train_AE))

        x_test = np.concatenate((x_test[:int(x_noise_test.shape[0])], x_noise_test))
        y_test_ROI = np.concatenate((y_test_signal_ROI[:int(x_noise_test.shape[0])], y_noise_test_ROI))
        y_test_AE = np.concatenate((y_test_AE[:int(x_noise_test.shape[0])], y_noise_test_AE))

        print('')
        print('     x_train (1DCNN + AE) : ', x_train.shape)
        print('     x_test  (1DCCN + AE) : ', x_test.shape)
        print('     y_train (1DCNN)      : ', y_train_ROI.shape)
        print('     y_test  (1DCNN)      : ', y_test_ROI.shape)
        print('     y_train (AE)         : ', y_train_AE.shape)
        print('     y_test  (AE)         : ', y_test_AE.shape)
        print('')

        # extra train shuffle for good measure 
        x_train, y_train_ROI, y_train_AE = shuffle(x_train,y_train_ROI, y_train_AE)
        # extra test shuffle for good measure 
        x_test, y_test_ROI, y_test_AE = shuffle(x_test,y_test_ROI, y_test_AE)
        print('--> shuffled train+test set for good measure')
        print('')

        outpath = "C:\\Users\\vanth\\Desktop\\Workspace\\AUTOENCODER\\processed_data_LATEST\\"

        np.save(outpath + "x_train_" + wireplane, x_train)
        print('--> x_train saved to', outpath + "x_train_" + wireplane)

        np.save(outpath + "x_test_" + wireplane, x_test)
        print('--> x_test saved to', outpath + "x_test_" + wireplane)

        np.save(outpath + "y_train_ROI_" + wireplane, y_train_ROI)
        print('--> y_train_1DCNN saved to', outpath + "y_train_ROI_" + wireplane)

        np.save(outpath + "y_train_AE_" + wireplane, y_train_AE)
        print('--> y_train_AE saved to', outpath + "y_train_AE_" + wireplane)
        
        np.save(outpath + "y_test_ROI_" + wireplane, y_test_ROI)
        print('--> y_test_1DCNN saved to', outpath + "y_test_ROI_" + wireplane)

        np.save(outpath + "y_test_AE_" + wireplane, y_test_AE)
        print('--> y_test_AE saved to', outpath + "y_test_AE_" + wireplane)
        print('')
    else:
        print("INPUT ERROR: python prepocess.py -plane [wireplane: U, V,Z]")


if __name__ == '__main__':
    main()
