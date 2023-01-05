# Physics Informed Neural Net (PINN) for a Spring-Mass-Damper system

This repo demonstrates how a physics informed neural net (PINN) can generalize to out of sample data using a simple 1-D spring-mass-damper system. This repo is based on an example by Ben Moseley [So, what is a physics-informed neural network?](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)

![Results with PINN](/Images/training_results_with_physics_informed.png)

---

## Environment and Packages

This repo was built with python version 3.6.15. The only package outside the standard packages required for this repo are:

* pytorch version 1.10.2
* numpy version 1.16.4.
* matplotlib version 3.3.2
* sklearn version 0.23.2

The [requirements.txt](requirements.txt) allows the user to create an environment with this package by running the command: python3 -m pip install -r requirements.txt

---

## Spring-Mass-Damper

 The spring-mass-damper system is implemented in [SpringMassDamper.py](SpringMassDamper.py) as a time-stepped PDE solver. At each time step, the program solves for the spring and damper force and then finds the acceleration, velocity, and position using the following equations.

acceleration(t+1) = acceleration(t) - (spring_force + damper_force) / mass

velocity(t+1) = velocity(t) + (acceleration(t+1) * self.dt

position(t+1) = position(t) + (velocity(t+1) * dt)

Using the system parameters:

* Mass = 1 kg
* Spring coefficient (k) = 2.5 N/m
* Damper coefficient (c) = 0.3 N/m/s

and the initial conditions:

* Position(0) = 15 m
* Velocity(0) = 0 m/s
* Acceleration(0) = 0 m/s^2

Results in the over damped oscillations below:

![Spring-mass-system](/Images/position_velocity_acceleration_history.png)

## Physics Informed Neural Net

The PINN is created by including the physics loss into the overall neural net's loss calculations. Normally, a neural net tries to minimize the difference between it's prediction (y_hat) and the true output (y) using mean squared error (MSE) or a similar metric.

MSE = (y_hat - y)^2

PINNs include additional terms into the loss function which can include the physics loss, boundary condition loss, and initial condition loss. In this example only the physics loss is explicitly calculated. The boundary loss is included in the MSE calculations. The spring-mass-damper is described by the partial differential equation:

acceleration(t) + DAMPING_COEFFICIENT * velocity(t) + SPRING_COEFFICIENT * position(t) = 0

The (auto differentiation) [https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html] function within most neural net packages allows the program to find the velocity (derivative of position) and accelerate (derivative of velocity) fairly easy. Thus, as the neural net is training, it can predict the position for a set of points during the forward pass. We then use auto differentiation to find the velocity and acceleration of those points.

This allows us to use the full loss function:

J = L_MSE + L_physics.

## Training Data

To show the benefits of PINNs, the training data is sampled from only half of the full 1-D space. The training data consists of 20 equally-spaced points between 0 and 0.5 seconds.

![TrainingData](/Images/training_data.png)

During the calculation of the physics loss for the PINN, an additional set of points are also calculated using only the forward pass of the neural network. These 500 points span from 0 to 1 second.

## Results

A neural net trained with only the MSE loss results in a good in-sample fit, but fails to generalize to new data as seen below.

![training_results_noPINN](/Images/training_results_standard_loss_function.png)

Using a PINN approach, the neural net is trained not only on the training data, but also on the physics of the problem. This results in a neural net that is able to extrapolate outside of the training data with a high degree of accuracy.

![training_results_withPINN](/Images/training_results_with_physics_informed.png)

---

## References

Moseley, Ben. "So, what is a physics-informed neural network?" 28 August 2021, [So, what is a physics-informed neural network?](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
