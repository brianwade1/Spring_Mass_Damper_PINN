"""
Creates a fully connected feed forward neural network. It includes methods 
to define, configure, train, and evaluate the model. 
Created 28 Oct 2022
@author: Brian Wade
@version: 1.0
"""
# Standard library imports
import math
import os
import pickle
import random
import datetime

# Conda imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, History
from tensorflow.keras.utils import plot_model


class NN_Regression_Model:
    def __init__(self, hidden_nodes, input_dim, output_dim, 
                learning_rate=1e-2, 
                loss_function='mean_absolute_error',
                seed=42):

        self.num_layers = len(hidden_nodes)
        self.layer_sizes = hidden_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.learning_rate_min = 1e-5
        self.loss_function = loss_function
        self.metrics = ["mse", "mae"]
        self.patience = 10
        self.min_delta = 1e-6
        self.validation_split = 0.2
        self.shuffle = False
        self.verbose = True
        self.max_epochs =  1000 
        self.batch_size = 4
        self.learning_rate_scheduler = 'linear_lr_dec'
        self.decay_rate = 0.90
        self.num_decays_during_training = 10
        self.history = History()
        self.seed = seed
        self.log_dir = 'Log'

        self.decay_step = round(self.max_epochs / self.num_decays_during_training)

    def make_model(self):
        "Creates the neural net model"
        # Make NN model
        self.model = Sequential()
        self.model.add(Dense(30, input_dim = 1, activation = 'tanh')) 
        self.model.add(Dense(30, activation = 'tanh'))
        self.model.add(Dense(30, activation = 'tanh'))
        self.model.add(Dense(30, activation = 'tanh'))
        self.model.add(Dense(1, activation='linear'))
        # # Input and hidden layers
        # for layer_num, layer_size in enumerate(self.layer_sizes):
        #     if layer_num == 0:
        #         self.model.add(Dense(layer_size, input_dim = self.input_dim, activation = 'tanh')) 
        #     else:
        #         self.model.add(Dense(layer_size, activation = 'tanh'))

        # # output layer - default linear activation for regression
        # self.model.add(Dense(self.output_dim, activation='linear'))
        
        # Compile the network 
        optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate)       
        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=self.metrics)

        return self.model.summary

    def createScaler_and_scale_data(self, x_train, x_test, save_scaler=True):
        # Neural nets need to scaled data - fit to train, apply to val and test.
        # Check shape (if single dimension assume 1 feature)
        if x_train.ndim == 1:
            self.x_train = x_train.reshape(-1,1)
        else:
            self.x_train = x_train

        if x_test.ndim == 1:
            self.x_test = x_test.reshape(-1,1)
        else:
            self.x_test = x_test

        self.std_scaler = preprocessing.StandardScaler()
        self.x_train_scaled = self.std_scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.std_scaler.transform(self.x_test)

        self.save_scaler = save_scaler
        if self.save_scaler:
            # Save scaler info for later deployment
            scaler_filename = os.path.join('Models', 'std_scaler.save')
            with open(scaler_filename, "wb") as f: 
                pickle.dump(self.std_scaler, f) 
        
        self.x_sets_scaled = {'train': self.x_train_scaled, 'test': self.x_test_scaled}
        return self.x_sets_scaled

    def scale_data(self, x_sets):
        ''' Apply an already created scaler '''
        self.x_sets_scaled = dict()
        for dataset, datavalues in self.x_sets.items():
            self.x_sets_scaled[dataset] = self.scaler.transform(datavalues)
        return self.x_sets_scaled

    def new_run_log_dir(self): 
        #log_dir = os.path.join('./log', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.log_dir): 
            os.makedirs(self.log_dir) 
        run_id = len([name for name in os.listdir(self.log_dir)]) 
        run_log_dir = os.path.join(self.log_dir, str(run_id)) 
        return run_log_dir

    def train_model(self, x_data, y_data, model_name):

        # es = EarlyStopping(monitor = 'val_loss', 
        #                    mode = 'min', 
        #                    verbose = self.verbose, 
        #                    patience = self.patience, 
        #                    min_delta = self.min_delta,
        #                    restore_best_weights = True)
                
        tnan = TerminateOnNaN()
        run_log_dir = self.new_run_log_dir()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1)

        full_model_name = os.path.join('Models', model_name)
        mc = ModelCheckpoint(full_model_name, 
                           monitor = 'val_loss', 
                           mode = 'min', 
                           verbose = self.verbose, 
                           save_best_only = True)

        def lr_step_power_scheduler(epoch, lr):
            if epoch % self.decay_step == 0 and epoch:
                return lr * pow(self.decay_rate, np.floor(epoch / self.decay_step))
            return lr

        def linear_lr_dec(epoch, lr):
            lr = self.learning_rate - ((self.learning_rate - self.learning_rate_min)/self.max_epochs)*epoch
            return lr

        def cosine_annealing(epoch, lr):
            current_max_lr = self.learning_rate - ((self.learning_rate - self.learning_rate_min) / self.max_epochs) * epoch 
                
            epochs_per_cycle = math.floor(self.max_epochs / self.num_decays_during_training)
            cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
            lr = current_max_lr/2 * (math.cos(cos_inner) + 1)
            return lr

        def cosine_annealing_linear(epoch, lr):
            if epoch <= self.max_epochs / 2:
                current_max_lr = self.learning_rate
            else: 
                current_max_lr = self.learning_rate_min + ((self.learning_rate - self.learning_rate_min)/(self.max_epochs - (self.max_epochs / 2)))*(self.max_epochs - epoch)

            epochs_per_cycle = math.floor(self.max_epochs / self.num_decays_during_training)
            cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
            lr = current_max_lr/2 * (math.cos(cos_inner) + 1)        
            return lr

        def no_lr_sched(epoch, lr):
            return lr

        if self.learning_rate_scheduler == 'lr_step_power_scheduler':
            lr_sched_method = lr_step_power_scheduler
        elif self.learning_rate_scheduler == 'linear_lr_dec':
            lr_sched_method = linear_lr_dec
        elif self.learning_rate_scheduler == 'cosine_annealing':
            lr_sched_method = cosine_annealing
        elif self.learning_rate_scheduler == 'cosine_annealing_linear':
            lr_sched_method = cosine_annealing_linear
        else:
            lr_sched_method = no_lr_sched
            
        lr_sched = LearningRateScheduler(lr_sched_method, verbose = self.verbose)

        x_train = x_data['train'].reshape(-1,1)
        y_train = y_data['train'].reshape(-1,1)
        self.history = self.model.fit(x_data['train'], y_data['train'],
                batch_size = self.batch_size, 
                epochs = self.max_epochs, 
                shuffle = self.shuffle,
                validation_split = self.validation_split, 
                callbacks = [tnan, mc, tensorboard_callback],
                verbose = self.verbose)
                #callbacks = [es, tnan, mc, lr_sched],
                #validation_data = (x_data['validation'], y_data['validation']),  
                #callbacks = [es, tnan, mc, lr_sched],

        pass

    def evaluate_model(self, x_sets, y_sets):
        self.y_hat = dict()
        self.MSE = dict()
        self.MAE= dict()
        for dataset, datavalues in x_sets.items():
            # predict output and predicted probability of positive class
            self.y_hat[dataset] = self.model.predict(datavalues)
            
            # log metrics in dict
            self.MSE[dataset] = mean_squared_error(y_sets[dataset], self.y_hat[dataset])
            self.MAE[dataset] = mean_absolute_error(y_sets[dataset], self.y_hat[dataset])
        
        return self.MSE, self.MAE, self.y_hat

    def save_model(self, file_name):
        full_file_name = os.path.join('Models', file_name)
        self.model.save(full_file_name)

    def load_model_sets(self, file_name):
        full_file_name = os.path.join('Models', file_name)
        self.model = load_model(full_file_name)
    
    def plot_model(self):
        '''Save image of model architecture'''
        plot_model(self.model, to_file = os.path.join('Images', 'NN_model.png'), show_shapes = True, show_layer_names = True)

    def save_model_summary(self):
        ''' Print model summary to file '''
        with open(os.path.join('Results', 'modelsummary.txt'),'w+') as f:
            self.model.summary(print_fn = lambda x: f.write(x + '\n'))

    def set_random_seeds(self):
        os.environ['PYTHONHASHSEED']=str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

    def record_results(self, MSE, MAE, results_filename):
        # save results
        with open(results_filename, 'a') as f:
            f.write('MSE' + '\n')
            for key, value in MSE.items():
                f.write(str(key) + ' = ' + str(value) + '\n')
            f.write('MAE' + '\n')
            for key, value in MAE.items():
                f.write(str(key) + ' = ' + str(value) + '\n')
            f.write('\n')
