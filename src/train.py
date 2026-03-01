"""Training pipeline for Parametric PINN.

Two-phase training:
  Phase 1: Adam optimizer — fast initial convergence
  Phase 2: L-BFGS — fine-grained physics-consistent optimization
"""

import tensorflow as tf
import numpy as np
import yaml
from src.model import ParametricPINN
from src.physics import navier_stokes_residuals


def compute_loss(model, x_pde, y_pde, Re_pde,
                 x_bc, y_bc, Re_bc, u_bc, v_bc,
                 weights):
    """Compute multi-component physics loss."""
    # PDE residuals
    r_c, r_mx, r_my = navier_stokes_residuals(model, x_pde, y_pde, Re_pde)
    loss_pde = tf.reduce_mean(r_c**2 + r_mx**2 + r_my**2)

    # Boundary condition loss
    inputs_bc = tf.stack([x_bc, y_bc, Re_bc], axis=1)
    uvp_bc = model(inputs_bc)
    loss_bc = tf.reduce_mean((uvp_bc[:, 0] - u_bc)**2 +
                             (uvp_bc[:, 1] - v_bc)**2)

    # Total weighted loss
    loss = weights['pde'] * loss_pde + weights['bc'] * loss_bc
    return loss, loss_pde, loss_bc


def train(config_path='configs/config.yaml'):
    """Main training function."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = ParametricPINN(
        layers=cfg['model']['hidden_layers'],
        neurons=cfg['model']['neurons_per_layer'],
        activation=cfg['model']['activation']
    )

    optimizer = tf.keras.optimizers.Adam(cfg['training']['learning_rate'])
    weights = {
        'pde': cfg['loss_weights']['lambda_pde'],
        'bc':  cfg['loss_weights']['lambda_bc']
    }

    print("Training started...")
    # TODO: Add data loading and full training loop
    # See notebooks/01_training.ipynb for complete walkthrough


if __name__ == '__main__':
    train()
