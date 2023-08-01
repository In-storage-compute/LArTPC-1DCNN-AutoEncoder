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
    np.random.seed(7)
    args = sys.argv[1:]
    planes_ = ['U', 'V', 'Z']
    batch_size_ = int(args[2])
    
    if len(args) == 3 and args[0] == '-plane' and args[1] in planes_:
        start_time = time.time()
        wireplane = args[1]
        path = '../processed_data/'
        
        with tf.device('/GPU:0'):
            x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, x_valid_scaled, y_valid_scaled, mean, std = funcs.load_data(path, wireplane)
            np.save('results/'+wireplane+'/models/mean_AE_' + wireplane, mean)
            np.save('results/'+wireplane+'/models/std_AE_' + wireplane, std)


            # Converting the numpy array to a tensor.
            #-------------------------------------------------------------------------
            def custom_mse2(y_true, y_pred):
                np_y_true = y_true.numpy()
                batch_size = batch_size_  # hard coded for now
                
                y_true_rescaled = np_y_true*std+mean

                sig_ranges = []
                #print('finding ranges where there are signals: ') 
                for i in range(len(y_true_rescaled)):
                    wave =  y_true_rescaled[i]
                    sig_ranges.append(funcs.merge_ranges(wave, 5))
                #print("finding ranges where there are no signals: ")
                no_sig_ranges = funcs.get_non_signal_ranges(sig_ranges)
                

                #for i in range(len(sig_ranges)):
                #    print('DEBUG MESSAGE: ',sig_ranges[i], '---', no_sig_ranges[i])

                #print('calculating MSEs')
                total_mse = 0
                for i in range(len(np_y_true)):
                        # total_mse += funcs.calculate_single_mse(np_y_true[i], np_y_pred[i], sig_ranges[i])
                        total_mse += funcs.calculate_single_mse(y_true[i], y_pred[i], sig_ranges[i], no_sig_ranges[i])
                
                loss = total_mse/batch_size_
                #print('batch loss:', loss.numpy())
                #batch_size
                #print('TESTTT:: --', alpha)
                #print('-ALPHA: ', int(int(alpha).numpy()))

                return loss

                #-------------------------------------------------------------------------------
            
            
            model = load_model('../ROI_models/model_' + wireplane + 'plane_nu_ROI.h5')
            autoencoder = Autoencoder(200, model, x_train_scaled)
            compiled_model = autoencoder.create_model()

            # see what happens when we unfreeze more layers of the 1dcnn
            for i,layer in enumerate(compiled_model.layers):                                      
                print(i,layer.name)
                
            # layer_num = 5 --> unfreeze last layer of 1dccnn and make params trainable
            # layer_nums = 6 --> all layers of 1dcnn are frozen and non-trainable
            layer_num = 6    
                
            for layer in compiled_model.layers[:layer_num]:                                               
                layer.trainable=False                                                          
            for layer in compiled_model.layers[layer_num:]:                                               
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
            
            
            earlystop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=10,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )

            history = compiled_model.fit(x_train_scaled,                                                              
                        y_train_scaled,                                                            
                        batch_size=batch_size_,                                              
                        epochs=100,                                                      
                        callbacks= [earlystop], #[NewCallback(alpha)], # callbacks=callbacks_list,
                        validation_data=(x_valid_scaled, y_valid_scaled),                                                                       
                        verbose=1)
            
            
                    
        compiled_model.save('results/'+wireplane+'/models/batch_size' + str(batch_size_) + '_' + wireplane + 'plane_nu.h5')
        total_time = int((time.time() - start_time)/60)
        plt.figure(figsize=(12, 8))                                                     
        plt.plot(history.history['loss'], "r--", label="Loss of training data", antialiased=True)
        plt.plot(history.history['val_loss'], "r", label="Loss of validation data", antialiased=True)
        plt.title('Model Loss (train time: ' + str(total_time) + ' mins)',fontsize=15)                                            
        plt.ylabel('Loss (MSE)', fontsize=12)                                                 
        plt.xlabel('Training Epoch', fontsize=12)                                                                                                                       
        plt.legend(fontsize=12)
        filename = 'results/'+ wireplane +'/loss_plots/batch_size' + str(batch_size_) + '_' + wireplane + '_loss.png'
        plt.savefig(filename, facecolor='w', bbox_inches='tight')
        plt.close()
        #plt.show()
         
        print("train time:", time.time() - start_time, "to run")     



    else:
        print("INPUT ERROR: python ae_model.py -plane [wireplane: U, V,Z]")

if __name__ == '__main__':
    main()


