"""Navier-Stokes PDE Residuals.

Computes physics residuals for the incompressible Navier-Stokes equations:
    Continuity: du/dx + dv/dy = 0
    Momentum-x: u*du/dx + v*du/dy + dp/dx - nu*(d²u/dx² + d²u/dy²) = 0
    Momentum-y: u*dv/dx + v*dv/dy + dp/dy - nu*(d²v/dx² + d²v/dy²) = 0
"""

import tensorflow as tf


def navier_stokes_residuals(model, x, y, Re):
    """Compute NS equation residuals using automatic differentiation.
    
    Args:
        model: ParametricPINN model
        x, y: Spatial coordinates (tensors)
        Re: Reynolds number (tensor)
    
    Returns:
        r_continuity, r_momentum_x, r_momentum_y: PDE residuals
    """
    nu = 1.0 / Re  # kinematic viscosity

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, y])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y])
            inputs = tf.stack([x, y, Re], axis=1)
            uvp = model(inputs)
            u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]

        u_x = tape1.gradient(u, x)
        u_y = tape1.gradient(u, y)
        v_x = tape1.gradient(v, x)
        v_y = tape1.gradient(v, y)
        p_x = tape1.gradient(p, x)
        p_y = tape1.gradient(p, y)

    u_xx = tape2.gradient(u_x, x)
    u_yy = tape2.gradient(u_y, y)
    v_xx = tape2.gradient(v_x, x)
    v_yy = tape2.gradient(v_y, y)

    r_continuity = u_x + v_y
    r_momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return r_continuity, r_momentum_x, r_momentum_y
