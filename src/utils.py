"""Utility functions for visualization and data handling."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_flow_field(x, y, u, v, p, Re, save_path=None):
    """Plot velocity magnitude and pressure field.
    
    Args:
        x, y: Grid coordinates
        u, v: Velocity components
        p: Pressure field
        Re: Reynolds number (for title)
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Parametric PINN — Flow Over Cylinder (Re={Re})',
                 fontsize=13, fontweight='bold')

    velocity_mag = np.sqrt(u**2 + v**2)

    fields = [velocity_mag, u, p]
    titles = ['Velocity Magnitude |U|', 'Streamwise Velocity u', 'Pressure p']
    cmaps  = ['viridis', 'RdBu_r', 'coolwarm']

    for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
        sc = ax.scatter(x, y, c=field, cmap=cmap, s=1)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def sample_domain(x_range, y_range, Re_range, n_points):
    """Sample random collocation points in the flow domain."""
    x  = np.random.uniform(*x_range,  (n_points,))
    y  = np.random.uniform(*y_range,  (n_points,))
    Re = np.random.uniform(*Re_range, (n_points,))
    return x.astype(np.float32), y.astype(np.float32), Re.astype(np.float32)
