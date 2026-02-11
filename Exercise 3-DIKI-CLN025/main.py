# -*- coding: utf-8 -*-

from DataPreprocessing import alignDcd2Ref
from Model import VAEmodel
from PosteriorReplan import init_estimator,iteratively_update
from Model import Sampling_Layer
from SimilarityPlot import simliarityplot

from tensorflow import keras
import numpy as np
import tensorflow as tf

# state if Cartesian coordinates is aligned
isaligned = True

# data preprocessing
## parameters of alignDcd2Ref
psf_file_path = 'cln025.psf'
dcd_file_path = 'cln025.dcd'
reference_pdb_file = 'cln025.pdb'
aligned_file = 'cln025_aligned.dcd'
aligned_npz = 'cln025_algined.npz'
selection = 'not (resname WAT) and not (resname Na+) and not (name H*)'


# state if warm-up is finished
iswarmedup = True

# warm-up hyperparameters
feature_number = [10,3]
layers_dim = [50,30,20,10]
warm_up_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
warmed_up_model_path = 'warmedup.h5'

## warm-up callbacks
earlystop = keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=500, mode='auto')

savebestmodel = keras.callbacks.ModelCheckpoint(warmed_up_model_path, 
                                                monitor = 'reconstruction_loss', 
                                                verbose = 1,                                                      
                                                save_best_only = True, 
                                                mode = 'auto')


# hyperparameters of iteratively update
min_cluster_size = 1000        # FFK
sigma = 0                      # CFK
min_samples = 100
cluster_selection_method = 'eom'
saved_path = 'replan_%d_%d_%d.h5'%(min_cluster_size,min_samples,sigma)
iter_update_optimizer = keras.optimizers.RMSprop(learning_rate=1e-4)
lam = 1.
batch_size = 128
epochs = 200

# plot hyperparameters
img_saved_path = 'similarity_heatmap.png'

if __name__ == '__main__':
    
    # align dcd 2 reference
    if isaligned:
        npz = np.load(aligned_npz)
        all_data = npz['dcd_position']
        npz.close()
        
    else:
        all_data = alignDcd2Ref(psf_file_path, dcd_file_path, reference_pdb_file,\
                                aligned_file, aligned_npz, selection)
            
    
    
    # warm-up stage
    ## load or train warm-up model
    if iswarmedup:
        model = keras.models.load_model(warmed_up_model_path,\
                                    custom_objects={'Sampling_Layer':Sampling_Layer})
        
    else:
        no_use = tf.zeros((tf.shape(all_data)[0],2))
        
        model = VAEmodel(feature_number,layers_dim)
        model.compile(loss=['mse','mse','mse'],
                      loss_weights=[1,0,0],
                      optimizer=warm_up_optimizer)
        
        ## model training
        warm_up_history = model.fit(all_data,[all_data,no_use,no_use],
                                     epochs=10000,
                                     steps_per_epoch=1000,
                                     batch_size=256,
                                     callbacks=[earlystop,savebestmodel])
    
    
    # iteratively update
    estimator = init_estimator(min_cluster_size,min_samples,sigma,cluster_selection_method)
    
    # Replan distribution
    iteratively_update(all_data, model, iter_update_optimizer, estimator, saved_path, lam, batch_size, epochs)
    
    # plot similarity figure
    simliarityplot(estimator,model,all_data,img_saved_path)