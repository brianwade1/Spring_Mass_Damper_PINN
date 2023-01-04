"""
Creates a fully connected feed forward neural network for a Physics 
Informed Neural Network (PINN) where the known physics of the problem 
are included in the loss function of the network. 

Created 28 Oct 2022
@author: Brian Wade
@version: 1.0
"""

# Standard library imports
import os
import pickle
import random

# Conda imports
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Local imports
from SpringMassDamper import SpringMassDamper
from NN_Regression_Model import NN_Regression_Model


class Simulator():
    def __init__(self, simulation, num_steps):
        self.simulation = simulation
        self.num_steps = num_steps
        
    def collect_data(self):
        self.data = self.simulation.collect_data(self.num_steps)
        self.pos_data = np.array(self.data[0]).astype(np.float32)
        self.vel_data = np.array(self.data[1]).astype(np.float32)
        self.acc_data = np.array(self.data[2]).astype(np.float32)
        self.time_data = np.array(self.data[3]).astype(np.float32)

    def make_training_locations(self, num_training_points, training_step_start=0, training_step_end=-1, equal_space_training=True):       
        if training_step_end < 0:
            training_step_end = self.num_steps

        if equal_space_training:
            train_locations = np.linspace(training_step_start, training_step_end, num_training_points)
        else:
            potential_points = np.linspace(training_step_start, training_step_end, self.simulation.dt)
            train_locations = random.sample(potential_points, num_training_points)

        return train_locations.astype(int)

    def make_training_data(self, num_training_points, training_step_start=0, training_step_end=-1, equal_space_training=True):
        if not hasattr(self, 'data'):
            self.collect_data()
        train_locations = self.make_training_locations(num_training_points, training_step_start, training_step_end, equal_space_training)
        #train_data = np.vstack((self.time_data[train_locations], self.pos_data[train_locations]))
        #return train_data
        return self.time_data[train_locations], self.pos_data[train_locations]

    def make_test_data(self):
        if not hasattr(self, 'data'):
            self.collect_data()
        return self.time_data, self.pos_data
        
    def plot_results(self, x_train, y_train, y_hat, save_plot=True, physics_informed=False):
        fig, ax = plt.subplots()
        ax.plot(self.time_data, self.pos_data, label='Exact Solution')
        ax.set_title('Lateral Position Over Time')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('position (m)')
        ax.scatter(x_train, y_train, label='Training Data')
        ax.plot(self.time_data, y_hat, label='Prediction')
        ax.legend()
        plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
        if save_plot:
            if physics_informed:
                plot_name = 'training_results_with_physics_informed.png'
            else:
                plot_name = 'training_results_standard_loss_function'
            plt.savefig(os.path.join('Images', plot_name))
        else:
            plt.show()

    def plot_training_data(self, x, y, x_data, y_data, save_plot=True):
        plt.figure()
        plt.plot(x, y, label="Exact solution")
        plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
        plt.legend()
        if save_plot:
            plt.savefig(os.path.join('Images', 'training_data.png'))
        else:
            plt.show()


class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fc_start = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fc_hidden = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fc_end = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fc_start(x)
        x = self.fc_hidden(x)
        x = self.fc_end(x)
        return x  


if __name__=="__main__":
    # Constants
    SEED = 42
    RESULTS_FILENAME = os.path.join('Results', 'model_development_results.csv')
    NUMBER_SAMPLE_POINTS = 20

    # Initial values
    INITIAL_POSITION = 1
    INITIAL_VELOCITY = 0
    INITIAL_ACCELERATION = 0
    
    # Sim constants
    TIME_STEPS = 500
    dt = 0.002 # sampling period in seconds
    MASS = 1 # mass
    SPRING_COEFFICIENT = 400 #2.5 # spring coefficient (k)
    DAMPING_COEFFICIENT = 4 #0.3 # damping coefficient (b)

    # Code
    spring_mass_damper = SpringMassDamper(position = INITIAL_POSITION, 
                                velocity = INITIAL_VELOCITY, 
                                acceleration = INITIAL_ACCELERATION, 
                                dt = dt, 
                                mass = MASS, 
                                k = SPRING_COEFFICIENT,
                                c = DAMPING_COEFFICIENT)

    simulator = Simulator(spring_mass_damper, TIME_STEPS)
    
    x_sets = {}
    y_sets = {}
    x_sets['train'], y_sets['train'] = simulator.make_training_data(NUMBER_SAMPLE_POINTS, training_step_start=0, training_step_end=200, equal_space_training=True)
    x_sets['test'], y_sets['test'] = simulator.make_test_data()
  
    x_data = torch.tensor(x_sets['train']).view(-1, 1)
    y_data = torch.tensor(y_sets['train']).view(-1, 1)
    print(x_data.shape, y_data.shape)

    x = torch.tensor(x_sets['test']).view(-1, 1)
    y = torch.tensor(y_sets['test']).view(-1, 1)
    print(x.shape, y.shape)

    simulator.plot_training_data(x, y, x_data, y_data, save_plot=True)


    # # make model    
    # input_dim = x_sets['train'].ndim
    # output_dim = y_sets['train'].ndim
    # hidden_nodes = [10, 30, 10]
    # nn_model = NN_Regression_Model(hidden_nodes, input_dim, output_dim, SEED)
    # nn_model.set_random_seeds() 
    # #x_sets_scaled = nn_model.createScaler_and_scale_data(x_train=x_sets['train'], x_test=x_sets['test'], save_scaler=True)
    # model_summary = nn_model.make_model()  
    # training_history = nn_model.train_model(x_sets, y_sets, 'Neural_Net')
    # MSE, MAE, y_hat = nn_model.evaluate_model(x_sets, y_sets)
    # nn_model.record_results(MSE, MAE, RESULTS_FILENAME)
    # simulator.plot_results(x_sets['train'], y_sets['train'], y_hat['test'], save_plot=False)
    

    # train standard neural network to fit training data
    torch.manual_seed(SEED)
    model = FCN(1,1,32,3)

     # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    current_loss = 0.0
    EPOCHS = 1500
    for epoch in range(EPOCHS):
        # zero optimizer gradients in memory
        optimizer.zero_grad()
        # forward pass
        y_hat = model(x_data)
        # calc loss function
        loss = loss_function(y_hat, y_data)
        # Back Prop - calculate gradients
        loss.backward()
        # update weights and biases with gradients
        optimizer.step()
        # Print statistics
        current_loss += loss.item()
        if (epoch % 50 == 0) or (epoch == EPOCHS-1):
          avg_train_loss = current_loss / 50
          print(f'Loss after mini-batch {epoch}: {current_loss:.3f}')
          current_loss = 0.0

    with torch.no_grad():
        y_hat_train = model(x_data).detach().numpy()
        y_hat_test = model(x).detach().numpy()

    simulator.plot_results(x_sets['train'], y_sets['train'], y_hat_test, save_plot=True, physics_informed=False)


    

    # Create physics test locations
    #x_physics = torch.linspace(0,20,60).view(-1,1).requires_grad_(True)
    x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)

    current_loss = 0.0
    current_data_loss = 0.0
    current_physics_loss = 0.0
    EPOCHS = 20000
    for epoch in range(EPOCHS):
        # zero optimizer gradients in memory
        optimizer.zero_grad()
        # forward pass
        y_hat = model(x_data)
        # calc loss function
        
        # compute data loss
        loss_data = loss_function(y_hat, y_data)

        # compute physics loss
        y_hat_physics = model(x_physics)
        dx = torch.autograd.grad(y_hat_physics, x_physics, torch.ones_like(y_hat_physics), create_graph=True)[0]   # computes dy/dx
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0] # computes d^2y/dx^2
        res_physics = dx2 + DAMPING_COEFFICIENT * dx + SPRING_COEFFICIENT * y_hat_physics
        loss_physics = (1e-4)*torch.mean(res_physics**2)

        # Total loss is data and physics losses
        loss = loss_data + loss_physics

        # Back Prop - calculate gradients
        loss.backward()
        # update weights and biases with gradients
        optimizer.step()
        # Print statistics
        current_loss += loss.item()
        current_data_loss += loss_data.item()
        current_physics_loss += loss_physics.item()
        if (epoch % 50 == 0) or (epoch == EPOCHS-1):
            avg_train_loss = current_loss / 50
            print(f'Loss after mini-batch {epoch}: {current_loss:.5f} | Data loss: {current_data_loss: .5f} | Physics loss: {current_physics_loss: .5f}')
            current_loss = 0.0
            current_data_loss = 0.0
            current_physics_loss = 0.0

    with torch.no_grad():
        y_hat_train = model(x_data).detach().numpy()
        y_hat_test = model(x).detach().numpy()

    simulator.plot_results(x_sets['train'], y_sets['train'], y_hat_test, save_plot=True, physics_informed=True)

    yh = model(x).detach()
    xp = x_physics.detach()

    #plot_result(x, y, x_data, y_data, yh)


    pass