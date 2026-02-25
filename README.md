# Parametric PINNs for Flow Over Cylinders
> AI-Accelerated CFD using Physics-Informed Neural Networks

Parametric PINN surrogate that generalizes across Reynolds number 
variations for external flow over cylinders — replacing expensive 
full CFD re-simulations with real-time inference.

## Results
- ~95% accuracy vs traditional CFD
- 60% reduction in computational time
- <50ms real-time inference on NVIDIA T4 GPU

## Problem
Traditional CFD requires full re-simulation for every new Reynolds 
number. This parametric PINN learns the underlying Navier-Stokes 
physics once and generalizes across Re variations with a single network.

## Method
- Navier-Stokes equations enforced via automatic differentiation
- Multi-loss architecture: PDE loss + BC loss + data loss + regularization
- Reynolds number embedded as parametric input to the network

## Tech Stack
Python · TensorFlow/Keras · NumPy · Matplotlib · GPU Computing (NVIDIA T4)

## Project Status
 In Progress — part of M.Tech research at IIST

## Author
Raghavendra M  
M.Tech Aerospace Engineering (Thermal & Propulsion) @ IIST  
 ragharit586@gmail.com  
[LinkedIn](www.linkedin.com/in/raghavendram-aero)
