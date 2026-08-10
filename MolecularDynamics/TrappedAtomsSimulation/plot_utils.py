"""
Module: plot_utils.py
Author: Paul Christ
Date: August 2026

Description:
This module provides robust visualization and analysis tools for N-body 
molecular dynamics simulations. It includes functions to analyze energy 
and angular momentum conservation, thermalization processes, and 
phase-space trajectories.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional

# =====================================================================
# 1. CONSERVATION LAWS (ENERGY & ANGULAR MOMENTUM)
# =====================================================================

def plot_energy_conservation(results: Dict[str, torch.Tensor]):
    """
    Plots the energy components and the relative total energy drift.
    Replaces the old 'plot_energy_and_error' function.
    """
    t = results['times'].cpu().numpy()
    E_kin = results['kinetic_energy'].cpu().numpy()
    E_trap = results.get('potential_energy_trap', results.get('potential_energy_harmonic', torch.zeros_like(results['times']))).cpu().numpy()
    E_pair = results.get('potential_energy_pair', torch.zeros_like(results['times'])).cpu().numpy()

    E_tot = E_kin + E_trap + E_pair
    E0 = E_tot[0]
    
    # Avoid division by zero if initial energy is extremely close to 0
    if np.abs(E0) < 1e-12:
        rel_error = np.abs(E_tot - E0)
        ylabel_error = r'Absolute deviation $|\Delta E|$'
    else:
        rel_error = np.abs((E_tot - E0) / E0)
        ylabel_error = r'Relative deviation $|\Delta E / E_0|$'

    # --- Plot 1: Energy Contributions ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogy(t, E_kin, label='Kinetic', lw=1.5, alpha=0.8)
    if np.any(np.abs(E_trap) > 1e-12):
        ax1.semilogy(t, E_trap, label='Trap', lw=1.5, alpha=0.8)
    if np.any(np.abs(E_pair) > 1e-12):
        ax1.semilogy(t, E_pair, label='Pair Interaction', lw=1.5, alpha=0.8)
        
    ax1.semilogy(t, E_tot, '--', color='black', lw=2, label='Total Energy')
    ax1.set_ylabel('Energy [dimensionless]')
    ax1.set_xlabel('Time [dimensionless]')
    ax1.legend(loc='best', framealpha=0.6)
    ax1.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Energy Drift ---
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.semilogy(t, rel_error, color='tab:blue', lw=1.5)
    ax3.axhline(np.max(rel_error), color='red', ls=':', lw=1.2, label=f'Max Error: {np.max(rel_error):.2e}')
    ax3.set_ylabel(ylabel_error)
    ax3.set_xlabel('Time [dimensionless]')
    ax3.legend(loc='best', framealpha=0.6)
    ax3.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_angular_momentum_conservation(results: Dict[str, torch.Tensor]):
    """
    Plots the 3D angular momentum components and their relative drift.
    """
    q = results['positions'].cpu().numpy()
    p = results['momenta'].cpu().numpy()
    t = results['times'].cpu().numpy()

    # Calculate L = q x p and sum over all particles
    L_vec = np.cross(q, p, axis=2).sum(axis=1)
    Lx, Ly, Lz = L_vec[:, 0], L_vec[:, 1], L_vec[:, 2]

    L0 = L_vec[0]
    rel_drift_x = np.abs((Lx - L0[0]) / (np.abs(L0[0]) + 1e-12))
    rel_drift_y = np.abs((Ly - L0[1]) / (np.abs(L0[1]) + 1e-12))
    rel_drift_z = np.abs((Lz - L0[2]) / (np.abs(L0[2]) + 1e-12))

    # --- Plot 1: Components ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, Lx, label='$L_x$', color='tab:blue')
    ax1.plot(t, Ly, label='$L_y$', color='tab:orange')
    ax1.plot(t, Lz, label='$L_z$', color='tab:green')
    ax1.set_ylabel('Angular Momentum [dimensionless]')
    ax1.set_xlabel('Time [dimensionless]')
    ax1.legend(loc='best', framealpha=0.6)
    ax1.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Relative Drift ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.semilogy(t, rel_drift_x, label=r'$|\Delta L_x / L_{x0}|$', color='tab:blue')
    ax2.semilogy(t, rel_drift_y, label=r'$|\Delta L_y / L_{y0}|$', color='tab:orange')
    ax2.semilogy(t, rel_drift_z, label=r'$|\Delta L_z / L_{z0}|$', color='tab:green')
    ax2.set_ylabel('Relative deviation')
    ax2.set_xlabel('Time [dimensionless]')
    ax2.legend(loc='best', framealpha=0.6)
    ax2.grid(True, which='both', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =====================================================================
# 2. PHASE SPACE & COLLISION ANALYSIS
# =====================================================================

def plot_zoomed_collision_with_circle(results: Dict[str, torch.Tensor], window_steps: int = 300):
    """
    Searches for the hardest collision (max delta p) and plots the phase 
    space of the two involved particles within a zoomed time window.
    """
    q_all = results['positions'].cpu().numpy()
    p_all = results['momenta'].cpu().numpy()

    # Find the hardest collision
    dp = np.diff(p_all, axis=0)
    dp_mag = np.linalg.norm(dp, axis=2)
    t_col, p1 = np.unravel_index(np.argmax(dp_mag), dp_mag.shape)

    # Find the collision partner
    q_at_collision = q_all[t_col, :, :]
    q_p1 = q_at_collision[p1, :]
    distances = np.linalg.norm(q_at_collision - q_p1, axis=1)
    distances[p1] = np.inf
    p2 = np.argmin(distances)

    particles = [p1, p2]
    labels = [f'Partner 1 (ID {p1})', f'Partner 2 (ID {p2})']
    colors = ['tab:red', 'tab:orange']
    dim_names = ['x', 'y', 'z']

    # Define the time window
    t_start = max(0, t_col - window_steps)
    t_end = min(q_all.shape[0], t_col + window_steps)

    q_pair_window = q_all[t_start:t_end, particles, :]
    p_pair_window = p_all[t_start:t_end, particles, :]
    q_margin = (q_pair_window.max() - q_pair_window.min()) * 0.1
    p_margin = (p_pair_window.max() - p_pair_window.min()) * 0.1
    
    global_xlim = (q_pair_window.min() - q_margin, q_pair_window.max() + q_margin)
    global_ylim = (p_pair_window.min() - p_margin, p_pair_window.max() + p_margin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for d in range(3):
        ax = axes[d]
        for idx, (p_idx, color, label) in enumerate(zip(particles, colors, labels)):
            q_d = q_all[t_start:t_end, p_idx, d]
            p_d = p_all[t_start:t_end, p_idx, d]
            
            ax.plot(q_d, p_d, lw=1.5, color=color, alpha=0.9, label=label)
            ax.scatter(q_d[0], p_d[0], color=color, marker='s', s=40, edgecolors='black', zorder=5)
            ax.scatter(q_all[t_col, p_idx, d], p_all[t_col, p_idx, d],
                       s=300, facecolors='none', edgecolors='black', lw=1.5, zorder=10)

        ax.set_xlabel(f'Position $q_{dim_names[d]}$ [dim.los]')
        if d == 0: 
            ax.set_ylabel(f'Momentum $p_{dim_names[d]}$ [dim.los]')
        ax.set_title(f'Phase Space {dim_names[d].upper()}-Component')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        if d == 0:
            ax.legend(loc='best', framealpha=0.8)
            
    plt.tight_layout()
    plt.show()


def plot_simple_momentum_histogram(
    p_tensor_dimless: torch.Tensor,
    bins: int = 75,
    title: str = "Momentum Distribution"
):
    p_magnitudes = torch.sqrt(torch.sum(p_tensor_dimless**2, dim=1)).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(p_magnitudes, bins=bins, density=True, alpha=0.8, label='Simulated Distribution')
    plt.title(title, fontsize=16)
    plt.xlabel("Dimensionless Momentum |p|")
    plt.ylabel("Probability Density")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================================================
# 3. THERMALIZATION & KINETIC ENERGY ANALYSIS
# =====================================================================

def calculate_single_group_temperature(
    momenta: torch.Tensor,
    n_particles: int,
    T_ref: float
) -> torch.Tensor:
    """
    Calculates the temperature for a single particle group.
    """
    E_kin_dimless_all = 0.5 * torch.sum(momenta**2, dim=(1, 2))
    T_all = T_ref * (2.0 / (3.0 * n_particles)) * E_kin_dimless_all
    return T_all


def plot_temperature_evolution(
    results_dict: Dict,
    n_particles: int,
    T_ref: float,
    T0_s: float  
):
    """
    Plots the evolution of the mean directional kinetic energy for a single 
    particle group over time.
    """
    T_all_K = calculate_single_group_temperature(
        results_dict['momenta'],
        n_particles,
        T_ref
    )

    # Allow T0_s to be either float or tensor for backwards compatibility
    if isinstance(T0_s, torch.Tensor):
        T0_s = T0_s.item()

    t_phys_ms = results_dict['times'].cpu().numpy() * T0_s * 1000
    T_all_uK = T_all_K.cpu().numpy() * 1e6

    plt.figure(figsize=(10, 5))
    plt.plot(t_phys_ms, T_all_uK, label=f"Total ({n_particles} Particles)", color='purple')
    plt.title("Kinetic Energy Evolution", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel(r"Mean Kinetic Energy per Dimension [$\mu K \cdot k_B$]")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


def calculate_group_temperatures(
    momenta: torch.Tensor,
    n_groups: Tuple[int, int],
    T_ref: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the temperature separately for two interacting groups."""
    n1 = n_groups[0]
    p_group1 = momenta[:, :n1, :]
    p_group2 = momenta[:, n1:, :]
    
    E_kin_dimless_group1 = 0.5 * torch.sum(p_group1**2, dim=(1, 2))
    E_kin_dimless_group2 = 0.5 * torch.sum(p_group2**2, dim=(1, 2))
    
    T_group1 = T_ref * (2.0 / (3.0 * n_groups[0])) * E_kin_dimless_group1
    T_group2 = T_ref * (2.0 / (3.0 * n_groups[1])) * E_kin_dimless_group2
    
    return T_group1, T_group2


def plot_thermalization(
    results: Dict,
    n_groups: Tuple[int, int],
    temp_groups: Tuple[float, float],
    T0_s: float  
):
    """Plots the thermalization curve for two distinct particle groups tracking mean kinetic energy."""
    T_ref = temp_groups[0]
    T_group1, T_group2 = calculate_group_temperatures(results['momenta'], n_groups, T_ref)
    
    if isinstance(T0_s, torch.Tensor):
        T0_s = T0_s.item()
        
    t_phys_ms = results['times'].cpu().numpy() * T0_s * 1000
    T_eq = (n_groups[0] * temp_groups[0] + n_groups[1] * temp_groups[1]) / sum(n_groups)

    plt.figure(figsize=(10, 5))
    plt.plot(t_phys_ms, T_group1.cpu().numpy() * 1e6, label=f"Group 1 ({n_groups[0]} Part.)", color='tab:blue')
    plt.plot(t_phys_ms, T_group2.cpu().numpy() * 1e6, label=f"Group 2 ({n_groups[1]} Part.)", color='tab:red')
    plt.axhline(y=T_eq * 1e6, color='black', linestyle='--', label=rf"$E_{{eq}} / k_B = {T_eq*1e6:.2f} \mu K$")
    
    plt.title("Thermalization of Particle Groups", fontsize=16)
    plt.xlabel("Time [ms]")
    plt.ylabel(r"Mean Kinetic Energy per Dimension [$\mu K \cdot k_B$]")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

# =====================================================================
# 4. POTENTIAL & DISTRIBUTION
# =====================================================================

def plot_potential_and_distribution_by_axis(
    positions: torch.Tensor,
    trap_force_func: Callable,
    trap_params: Dict,
    title_suffix: str = "",
    xyz_lim: float = 100.0,
    bins: int = 100
):
    """
    Plots the trap potential alongside the particle density histogram 
    along the x-, y-, and z-axes.
    """
    # --- 1. Prepare Data ---
    L0_tensor = trap_params.get('L0', 1.0)
    E0_tensor = trap_params.get('E0', 1.0)

    L0 = L0_tensor.item() if isinstance(L0_tensor, torch.Tensor) else L0_tensor
    E0 = E0_tensor.item() if isinstance(E0_tensor, torch.Tensor) else E0_tensor
    kB = 1.380649e-23

    device = positions.device
    precision = positions.dtype

    # Positions for histogram (in µm)
    q_cpu = positions.detach().cpu().numpy()
    q_x_um = q_cpu[:, 0] * L0 * 1e6
    q_y_um = q_cpu[:, 1] * L0 * 1e6
    q_z_um = q_cpu[:, 2] * L0 * 1e6

    # --- 2. Calculate Potential Curves ---
    N_plot_points = 200
    
    # Generate dimensionless positions for plot (units of L0)
    plot_x = torch.linspace(-xyz_lim / (L0 * 1e6), xyz_lim / (L0 * 1e6), N_plot_points, device=device, dtype=precision)
    plot_y = torch.linspace(-xyz_lim / (L0 * 1e6), xyz_lim / (L0 * 1e6), N_plot_points, device=device, dtype=precision)
    plot_z = torch.linspace(-xyz_lim / (L0 * 1e6), xyz_lim / (L0 * 1e6), N_plot_points, device=device, dtype=precision)

    # Create coordinate vectors along the axes
    zeros = torch.zeros(N_plot_points, device=device, dtype=precision)
    q_probe_x_axis = torch.stack([plot_x, zeros, zeros], dim=1)
    q_probe_y_axis = torch.stack([zeros, plot_y, zeros], dim=1)
    q_probe_z_axis = torch.stack([zeros, zeros, plot_z], dim=1)

    # Calculate Potentials
    _f, _pt, pot_x_dimless = trap_force_func(q_probe_x_axis, **trap_params)
    _f, _pt, pot_y_dimless = trap_force_func(q_probe_y_axis, **trap_params)
    _f, _pt, pot_z_dimless = trap_force_func(q_probe_z_axis, **trap_params)

    # Convert potential to µK and axes to µm for plots
    pot_x_uK = pot_x_dimless.detach().cpu().numpy() * E0 / kB * 1e6
    pot_y_uK = pot_y_dimless.detach().cpu().numpy() * E0 / kB * 1e6
    pot_z_uK = pot_z_dimless.detach().cpu().numpy() * E0 / kB * 1e6

    plot_x_um = plot_x.detach().cpu().numpy() * L0 * 1e6
    plot_y_um = plot_y.detach().cpu().numpy() * L0 * 1e6
    plot_z_um = plot_z.detach().cpu().numpy() * L0 * 1e6

    # --- 3. Plotting ---
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    if title_suffix:
        fig.suptitle(f"Potential and Particle Distribution: {title_suffix}", fontsize=16)

    # X-Axis
    ax[0].hist(q_x_um, bins=bins, density=True, alpha=0.7, label='Density')
    ax[0].set_xlabel("x-Position [µm]")
    ax[0].set_ylabel("Density [a.u.]")
    ax_pot_0 = ax[0].twinx()
    ax_pot_0.plot(plot_x_um, pot_x_uK, 'r-', lw=2, label='Potential U(x,0,0)')
    ax_pot_0.set_ylabel("Potential [µK]")
    ax[0].set_title("X-Axis")
    ax[0].legend(loc='upper left')
    ax_pot_0.legend(loc='upper right')

    # Y-Axis
    ax[1].hist(q_y_um, bins=bins, density=True, alpha=0.7, label='Density')
    ax[1].set_xlabel("y-Position [µm]")
    ax[1].set_ylabel("Density [a.u.]")
    ax_pot_1 = ax[1].twinx()
    ax_pot_1.plot(plot_y_um, pot_y_uK, 'r-', lw=2, label='Potential U(0,y,0)')
    ax_pot_1.set_ylabel("Potential [µK]")
    ax[1].set_title("Y-Axis")

    # Z-Axis
    ax[2].hist(q_z_um, bins=bins, density=True, alpha=0.7, label='Density')
    ax[2].set_xlabel("z-Position [µm]")
    ax[2].set_ylabel("Density [a.u.]")
    ax_pot_2 = ax[2].twinx()
    ax_pot_2.plot(plot_z_um, pot_z_uK, 'r-', lw=2, label='Potential U(0,0,z)')
    ax_pot_2.set_ylabel("Potential [µK]")
    ax[2].set_title("Z-Axis")

    plt.tight_layout()
    plt.show()
