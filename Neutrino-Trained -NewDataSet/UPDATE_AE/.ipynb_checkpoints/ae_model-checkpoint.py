"""
Van Tha Bik Lian
Jan. 06, 2023
train autoencoder using 1DCNN roi finder
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tqdm
import Custom_MSE as funcs
import pandas as pd
import time
from keras import backend as K
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense,Flatten, AveragePooling1D
from tensorflow.keras.layers import Input,  UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
tf.config.list_physical_devices('GPU')

class Autoencoder:
    def __init__(self, time_periods, cnn_model, x_train_scaled):
        self.time_periods = time_periods
        self.input_wave = Input(shape=(x_train_scaled.shape[1], 1))
        self.cnn_model = cnn_model
        
    def create_model(self):
        x = Conv1D(filters=16, kernel_size=3, strides=2, activation = "relu", 
                        weights=self.cnn_model.layers[0].get_weights() , input_shape=(self.time_periods,1))(self.input_wave)
        x = MaxPooling1D(pool_size=2)(x)
        
        #second convolutional block
        x = Conv1D(filters=32, kernel_size=5, strides=2, weights=self.cnn_model.layers[2].get_weights(), activation = "relu", )(x)
        x = MaxPooling1D(pool_size=2)(x)

        encoded = Conv1D(filters=64, kernel_size=9, weights=self.cnn_model.layers[5].get_weights(), activation = "relu")(x)
        
        x = Conv1D(filters=64, kernel_size=9, padding = "same", activation = "relu")(encoded)
        x = UpSampling1D(4)(x)
        x = Conv1D(filters=32, kernel_size=5, padding = "same", activation = "relu")(x)
        x = UpSampling1D(3)(x)
        x = Conv1D(filters=16, kernel_size=3,  activation = "relu")(x)
        x = UpSampling1D(6)(x)
        decoded = Conv1D(filters=1, kernel_size=5,  activation = "linear")(x)
        return Model(self.input_wave, decoded)
    

def main():
    np.random.seed(42)
    args = sys.argv[1:]
    planes_ = ['U', 'V', 'Z']
    
    if len(args) == 2 and args[0] == '-plane' and args[1] in planes_:
        start_time = time.time()
        wireplane = args[1]
        path = '../processed_data/current/'
        
        with tf.device('/GPU:0'):
            x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, mean, std = funcs.load_data(path, wireplane)

            y_true_rescaled = []
            print('rescalling y_true...')
            for i in tqdm.trange(len(y_train_scaled)):
                if sum(y_train_scaled[i]) == 0:
                    y_true_rescaled.append(y_train_scaled[i])
                else:
                    y_true_rescaled.append((std*y_train_scaled[i]+mean))
            y_true_rescaled = np.array(y_true_rescaled)

            sig_ranges = []
            print('finding ranges where there are signals: ') 
            for i in tqdm.trange(len(y_true_rescaled)):
                wave =  y_true_rescaled[i]
                sig_ranges.append(funcs.merge_ranges(wave, 5))
            print("finding ranges where there are no signals: ")
            no_sig_ranges = funcs.get_non_signal_ranges(sig_ranges)

            # Converting the numpy array to a tensor.
            #-------------------------------------------------------------------------
            def custom_mse2(y_true, y_pred):
                print('DEBUG - sig_ranges_len: ' + str(len(sig_ranges)))
                np_y_true = y_true.numpy()
                batch_size = 2048  # hard coded for now
                
                batchIdx = int(int(alpha).numpy())
                left_idx = batchIdx*batch_size
                loop_len = 2048
                print("BATCH IDX: " + str(left_idx))
                curr_sig_ranges = sig_ranges[left_idx:]
                curr_no_sig_ranges = no_sig_ranges[left_idx:]

                for i in range(10):
                    print('DEBUG MESSAGE: ',curr_sig_ranges[i], '---', curr_no_sig_ranges[i])

                print('calculating MSEs')
                total_mse = 0
                print('np_true len: ' + str(len(np_y_true)), np_y_true.shape)
                
                if batchIdx == 39:
                    for idx in tqdm.trange(128):
                        if sum(np_y_true[idx]) == 0:
                            # total_mse += funcs.calculate_single_mse_helper(np_y_true[i], np_y_pred[i])
                            total_mse += funcs.calculate_single_mse_helper(y_true[idx], y_pred[idx])
                            # total_mse += 0.3*funcs.calculate_single_mse_helper(y_true[i], y_pred[i])

                        else:
                            # total_mse += funcs.calculate_single_mse(np_y_true[i], np_y_pred[i], sig_ranges[i])
                            total_mse += funcs.calculate_single_mse(y_true[idx], y_pred[idx], curr_sig_ranges[idx], curr_no_sig_ranges[idx])
                            batch_size = 128
                    
                else:
                    print('else case')
                    for idx in tqdm.trange(loop_len):
                        if sum(np_y_true[idx]) == 0:
                            # total_mse += funcs.calculate_single_mse_helper(np_y_true[i], np_y_pred[i])
                            total_mse += funcs.calculate_single_mse_helper(y_true[idx], y_pred[idx])
                            # total_mse += 0.3*funcs.calculate_single_mse_helper(y_true[i], y_pred[i])

                        else:
                            # total_mse += funcs.calculate_single_mse(np_y_true[i], np_y_pred[i], sig_ranges[i])
                            total_mse += funcs.calculate_single_mse(y_true[idx], y_pred[idx], curr_sig_ranges[idx], curr_no_sig_ranges[idx])
                
                loss = total_mse/batch_size

                return loss

                #-------------------------------------------------------------------------------
            
            
            model = load_model('../latest_models/model_' + wireplane + 'plane_nu.h5')
            autoencoder = Autoencoder(200, model, x_train_scaled)
            compiled_model = autoencoder.create_model()

            # see what happens when we unfreeze more layers of the 1dcnn
            for i,layer in enumerate(compiled_model.layers):                                      
                print(i,layer.name)
                
            for layer in compiled_model.layers[:6]:                                               
                layer.trainable=False                                                          
            for layer in compiled_model.layers[6:]:                                               
                layer.trainable=True                                                           
            compiled_model.compile(optimizer='adam', loss=custom_mse2, run_eagerly=True)
            compiled_model.summary()

            alpha = K.variable(0)
            class NewCallback(keras.callbacks.Callback):
                def __init__(self, alpha):
                    self.alpha = alpha       
                def on_train_batch_begin(self, batch, logs={}):
                    K.set_value(self.alpha, batch)
                
                def on_epoch_begin(self, epochs, logs={}):
                    K.set_value(self.alpha, 0)
                

            print('-----------TRAINING STARTING NOW----------------')

            x_train_, x_valid, y_train_, y_valid =  train_test_split(x_train_scaled, y_test_scaled, 
                                                                     test_size=0.2, shuffle=False)
            
            sig_ranges = sig_ranges[:len(x_train_)]
            no_sig_ranges = no_sig_ranges[:len(x_train_)]

            earlystop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=3,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )
            
            history = compiled_model.fit(x_train_,                                                              
                        y_train_,                                                            
                        batch_size=2048,                                              
                        epochs=75,                                                      
                        callbacks=[NewCallback(alpha), earlystop], # callbacks=callbacks_list,
                        validation_data=(x_valid, y_valid),                                                               
                        verbose=1)
            
            
                    
        compiled_model.save("TESSSSSSSSSST" + wireplane + "plane_nu.h5")


        plt.figure(figsize=(12, 8))                                                     
        plt.plot(history.history['loss'], "r--", label="Loss of training data", antialiased=True)
        plt.plot(history.history['val_loss'], "r", label="Loss of validation data", antialiased=True)
        plt.title('Model Loss',fontsize=15)                                            
        plt.ylabel('Loss (MSE)', fontsize=12)                                                 
        plt.xlabel('Training Epoch', fontsize=12)                                                                                                                       
        plt.legend(fontsize=12)
        filename = 'TESSSSSSSSSSS-w2_dot7' + wireplane + '_loss.png'
        plt.savefig(filename, facecolor='w', bbox_inches='tight')
        plt.close()
        print("train time:", time.time() - start_time, "to run")     



    else:
        print("INPUT ERROR: python ae_model.py -plane [wireplane: U, V,Z]")

if __name__ == '__main__':
    main()