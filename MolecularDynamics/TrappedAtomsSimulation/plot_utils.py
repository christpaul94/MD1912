import torch
from typing import Dict, Tuple, Callable
import matplotlib.pyplot as plt

def plot_energy_and_error(t, kinetic_energy, potential_pl, potential_lj):
    """
    Plots:
    1. Energy contributions (linear scale)
    2. Energy contributions (log scale)
    3. Relative total energy deviation (log scale)
    All in subplots.
    """
    energy_contributions = [kinetic_energy]
    if potential_pl is not None:
        energy_contributions.append(potential_pl)

    if potential_lj is not None:
        energy_contributions.append(potential_lj)

    # Gesamtenergie und Fehler
    E_total = sum(energy_contributions)
    E0 = E_total[0]
    delta_E_rel = torch.abs(E_total - E0) / torch.abs(E0)

    t_np = t.cpu().numpy()

    # Subplots (3 Zeilen, 1 Spalte)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Lineare Skala
    axes[0].plot(t_np, kinetic_energy.cpu().numpy(), label='Kinetic Energy')
    if potential_pl is not None:
        axes[0].plot(t_np, potential_pl.cpu().numpy(), label='Trap Potential')

    if potential_lj is not None:
        axes[0].plot(t_np, potential_lj.cpu().numpy(), label='Interaction potential')
    axes[0].plot(t_np, E_total.cpu().numpy(), '--', color='r', label='Total Energy')
    axes[0].set_title('Energy Contributions (Linear Scale)')
    axes[0].set_ylabel(r'Energy')
    #axes[0].set_ylim(0,)
    axes[0].legend()
    axes[0].grid()

    # 2. Logarithmische Skala
    axes[1].plot(t_np, torch.abs(kinetic_energy).cpu().numpy(), label='Kinetic Energy')
    if potential_pl is not None:
        axes[1].plot(t_np, torch.abs(potential_pl).cpu().numpy(), label='Trap Potential')

    if potential_lj is not None:
        axes[1].plot(t_np, torch.abs(potential_lj).cpu().numpy(), label='Interaction potential')
    axes[1].plot(t_np, torch.abs(E_total).cpu().numpy(), '--', color='r', label='Total Energy')
    axes[1].set_yscale('log')
    axes[1].set_title('Energy Contributions (Logarithmic Scale)')
    axes[1].set_ylabel(r'$\log|E|$')
    axes[1].legend()
    axes[1].grid()

    # 3. Relative Energieabweichung
    axes[2].plot(t_np, delta_E_rel.cpu().numpy(), label='Relative Energy Deviation')
    axes[2].axhline(y=delta_E_rel.max().cpu(), linestyle=':', color='b', label=f'Max Error = {delta_E_rel.max():.2e}')
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-18, 1)
    axes[2].set_title('Energy Conservation Over Time')
    axes[2].set_xlabel(r'Time')
    axes[2].set_ylabel(r'$\Delta E / E_0$')
    axes[2].legend()
    axes[2].grid()

    # Layout anpassen
    plt.tight_layout()
    plt.show()

    return delta_E_rel.max()








def plot_simple_momentum_histogram(
    p_tensor_dimless: torch.Tensor,
    bins: int = 75,
    title: str = "Momentum Distribution"
):

    p_magnitudes = torch.sqrt(torch.sum(p_tensor_dimless**2, dim=1))

    p_magnitudes_np = p_magnitudes.cpu().numpy()

    plt.figure(figsize=(10, 6))

    plt.hist(p_magnitudes_np, bins=bins, density=True, alpha=0.8, label='Simulated Distribution')

    plt.title(title, fontsize=16)
    plt.xlabel("Dimensionsloser Impulsbetrag |p|")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()

def calculate_single_group_temperature(
    momenta: torch.Tensor,
    n_particles: int,
    T_ref: float
) -> torch.Tensor:
    """
    Berechnet die Temperatur für eine einzelne Gruppe (alle Teilchen).
    momenta:  
    n_particles: Gesamtzahl der Teilchen 
    T_ref: Referenztemperatur T 
    """
    # Summiere die kinetische Energie über Teilchen (dim 1) und Dimensionen (dim 2)
    # Das Ergebnis ist die dimless kinetische Energie für jeden Zeitschritt
    E_kin_dimless_all = 0.5 * torch.sum(momenta**2, dim=(1, 2))
    # T = T_ref * (2/3 * E_kin) / (N * k_B)
    T_all = T_ref * (2.0 / (3.0 * n_particles)) * E_kin_dimless_all

    return T_all


def plot_temperature_evolution(
    results_dict: Dict, #  Dictionary, das 'times' und 'momenta' enthält
    n_particles: int,
    T_ref: float,
    T0_s: torch.Tensor  
):
    """
    Plottet die Temperaturentwicklung für eine einzelne Teilchengruppe.
    """
    # Berechne die Temperatur über die Zeit
    T_all_K = calculate_single_group_temperature(
        results_dict['momenta'],
        n_particles,
        T_ref
    )

    # Konvertiere Zeiten in Millisekunden
    t_phys_ms = results_dict['times'].cpu().numpy() * T0_s.item() * 1000

    # Konvertiere Temperatur in Mikrokelvin für den Plot
    T_all_uK = T_all_K.cpu().numpy() * 1e6

    plt.figure(figsize=(12, 7))
    plt.plot(t_phys_ms, T_all_uK, label=f"Gesamttemperatur ({n_particles} Teilchen)", color='purple')

    plt.title("Temperaturentwicklung (Alle Teilchen)", fontsize=16)
    plt.xlabel("Zeit (ms)")
    plt.ylabel("Temperatur (µK)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

def calculate_group_temperatures(
    momenta: torch.Tensor,
    n_groups: Tuple[int, int],
    T_ref: float
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    T0_s: torch.Tensor  
):
    T_ref = temp_groups[0]
    T_group1, T_group2 = calculate_group_temperatures(results['momenta'], n_groups, T_ref)
    t_phys_ms = results['times'].cpu().numpy() * T0_s.item() * 1000
    T_eq = (n_groups[0] * temp_groups[0] + n_groups[1] * temp_groups[1]) / sum(n_groups)

    plt.figure(figsize=(12, 7))
    plt.plot(t_phys_ms, T_group1.cpu().numpy() * 1e6, label=f"Gruppe 1 ({n_groups[0]} Teilchen)", color='blue')
    plt.plot(t_phys_ms, T_group2.cpu().numpy() * 1e6, label=f"Gruppe 2 ({n_groups[1]} Teilchen)", color='red')
    plt.axhline(y=T_eq * 1e6, color='black', linestyle='--', label=f"T_eq = {T_eq*1e6:.2f} µK")
    plt.title("Thermalisierung der Teilchengruppen", fontsize=16)
    plt.xlabel("Zeit (ms)")
    plt.ylabel("Temperatur (µK)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()





import matplotlib.pyplot as plt
import numpy as np

def plot_potential_and_distribution_by_axis(
    positions: torch.Tensor,
    trap_force_func: Callable,
    trap_params: Dict,
    title_suffix: str = "",
    xyz_lim: float = 100.0,
    bins: int = 100
):
    """
    Plottet das Fallenpotential und die Teilchenverteilung (Histogramm)
    entlang der x-, y- und z-Achse.
    """

# --- 1. Daten vorbereiten ---


    L0_tensor = trap_params['L0']
    E0_tensor = trap_params['E0']


    L0 = L0_tensor.item() if isinstance(L0_tensor, torch.Tensor) else L0_tensor
    E0 = E0_tensor.item() if isinstance(E0_tensor, torch.Tensor) else E0_tensor

    kB = 1.380649e-23

    device = positions.device
    precision = positions.dtype

    # Positionen für Histogramm (in µm)
    q_cpu = positions.detach().cpu().numpy()
    q_x_um = q_cpu[:, 0] * L0 * 1e6
    q_y_um = q_cpu[:, 1] * L0 * 1e6
    q_z_um = q_cpu[:, 2] * L0 * 1e6

    # --- 2. Potentialverlauf berechnen ---
 
    N_plot_points = 200
    x_lim = xyz_lim
    y_lim = xyz_lim
    z_lim = xyz_lim

    # Erzeuge dimensionslose Postions für Plot (in Einheiten von L0)
    plot_x = torch.linspace(-x_lim / (L0 * 1e6), x_lim / (L0 * 1e6), N_plot_points, device=device, dtype=precision)
    plot_y = torch.linspace(-y_lim / (L0 * 1e6), y_lim / (L0 * 1e6), N_plot_points, device=device, dtype=precision)
    plot_z = torch.linspace(-z_lim / (L0 * 1e6), z_lim / (L0 * 1e6), N_plot_points, device=device, dtype=precision)

    # Erstelle Postions (entlang der Achsen)
    zeros = torch.zeros(N_plot_points, device=device, dtype=precision)
    q_probe_x_axis = torch.stack([plot_x, zeros, zeros], dim=1)
    q_probe_y_axis = torch.stack([zeros, plot_y, zeros], dim=1)
    q_probe_z_axis = torch.stack([zeros, zeros, plot_z], dim=1)

    # Berechne Potentiale
    _f, _pt, pot_x_dimless = trap_force_func(q_probe_x_axis, **trap_params)
    _f, _pt, pot_y_dimless = trap_force_func(q_probe_y_axis, **trap_params)
    _f, _pt, pot_z_dimless = trap_force_func(q_probe_z_axis, **trap_params)

    # Konvertiere Potential in µK und Achsen in µm für Plots
    pot_x_uK = pot_x_dimless.detach().cpu().numpy() * E0 / kB * 1e6
    pot_y_uK = pot_y_dimless.detach().cpu().numpy() * E0 / kB * 1e6
    pot_z_uK = pot_z_dimless.detach().cpu().numpy() * E0 / kB * 1e6

    plot_x_um = plot_x.detach().cpu().numpy() * L0 * 1e6
    plot_y_um = plot_y.detach().cpu().numpy() * L0 * 1e6
    plot_z_um = plot_z.detach().cpu().numpy() * L0 * 1e6

    # --- 3. Plotten ---
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Potential und Teilchenverteilung: {title_suffix}", fontsize=16)

    # --- X-Achse ---
    ax[0].hist(q_x_um, bins=bins, density=True, alpha=0.7, label='Teilchendichte')
    ax[0].set_xlabel("x-Position (µm)")
    ax[0].set_ylabel("Dichte (a.u.)")
    ax_pot_0 = ax[0].twinx() # Zweite y-Achse
    ax_pot_0.plot(plot_x_um, pot_x_uK, 'r-', label='Potential U(x,0,0)')
    ax_pot_0.set_ylabel("Potential (µK)")
    ax[0].set_title("X-Achse")
    ax[0].legend(loc='upper left')
    ax_pot_0.legend(loc='upper right')

    # --- Y-Achse ---
    ax[1].hist(q_y_um, bins=bins, density=True, alpha=0.7, label='Teilchendichte')
    ax[1].set_xlabel("y-Position (µm)")
    ax[1].set_ylabel("Dichte (a.u.)")
    ax_pot_1 = ax[1].twinx()
    ax_pot_1.plot(plot_y_um, pot_y_uK, 'r-', label='Potential U(0,y,0)')
    ax_pot_1.set_ylabel("Potential (µK)")
    ax[1].set_title("Y-Achse")

    # --- Z-Achse ---
    ax[2].hist(q_z_um, bins=bins, density=True, alpha=0.7, label='Teilchendichte')
    ax[2].set_xlabel("z-Position (µm)")
    ax[2].set_ylabel("Dichte (a.u.)")
    ax_pot_2 = ax[2].twinx()
    ax_pot_2.plot(plot_z_um, pot_z_uK, 'r-', label='Potential U(0,0,z)')
    ax_pot_2.set_ylabel("Potential (µK)")
    ax[2].set_title("Z-Achse")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für Haupttitel
    plt.show()
