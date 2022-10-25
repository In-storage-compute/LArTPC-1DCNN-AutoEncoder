Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 200, 1)]          0         
                                                                 
 conv1d (Conv1D)             (None, 99, 16)            64        
                                                                 
 max_pooling1d (MaxPooling1D  (None, 49, 16)           0         
 )                                                               
                                                                 
 conv1d_1 (Conv1D)           (None, 23, 32)            2592      
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 11, 32)           0         
 1D)                                                             
                                                                 
 conv1d_2 (Conv1D)           (None, 3, 64)             18496     
                                                                 
 conv1d_3 (Conv1D)           (None, 3, 64)             36928     
                                                                 
 up_sampling1d (UpSampling1D  (None, 18, 64)           0         
 )                                                               
                                                                 
 conv1d_4 (Conv1D)           (None, 18, 64)            32832     
                                                                 
 up_sampling1d_1 (UpSampling  (None, 72, 64)           0         
 1D)                                                             
                                                                 
 conv1d_5 (Conv1D)           (None, 66, 32)            14368     
                                                                 
 up_sampling1d_2 (UpSampling  (None, 264, 32)          0         
 1D)                                                             
                                                                 
 conv1d_6 (Conv1D)           (None, 258, 16)           3600      
                                                                 
 up_sampling1d_3 (UpSampling  (None, 1032, 16)         0         
 1D)                                                             
                                                                 
 conv1d_7 (Conv1D)           (None, 1025, 8)           1032      
                                                                 
 up_sampling1d_4 (UpSampling  (None, 4100, 8)          0         
 1D)                                                             
                                                                 
 conv1d_8 (Conv1D)           (None, 4092, 1)           73        
                                                                 
=================================================================
Total params: 109,985
Trainable params: 88,833
Non-trainable params: 21,152
_________________________________________________________________


ValueError: in user code:

    File "c:\Users\vanth\miniconda3\envs\tf\lib\site-packages\keras\engine\training.py", line 1160, in train_function  *
        return step_function(self, iterator)
    File "c:\Users\vanth\miniconda3\envs\tf\lib\site-packages\keras\engine\training.py", line 1146, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "c:\Users\vanth\miniconda3\envs\tf\lib\site-packages\keras\engine\training.py", line 1135, in run_step  **
...
    File "c:\Users\vanth\miniconda3\envs\tf\lib\site-packages\keras\losses.py", line 1486, in mean_squared_error
        return backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)

    ValueError: Dimensions must be equal, but are 4092 and 200 for '{{node mean_squared_error/SquaredDifference}} = SquaredDifference[T=DT_FLOAT](mean_squared_error/remove_squeezable_dimensions/Squeeze, IteratorGetNext:1)' with input shapes: [?,4092], [?,200].
