
import torch
import math
from typing import Dict, Tuple

kB = 1.380649e-23  # J/K

def initialize_momenta_thermal(
    n_particles: int, temperature: float, mass: float, kB: float, precision_type: torch.dtype
) -> torch.Tensor:
    """Initialisiert Impulse aus einer Maxwell-Boltzmann-Verteilung."""
    sigma_p = (mass * kB * temperature)**0.5
    momenta = torch.randn(n_particles, 3, dtype=precision_type) * sigma_p
    if n_particles > 1:
        momenta -= torch.mean(momenta, dim=0) # Gesamtimpuls auf Null setzen
    return momenta


def initialize_one_temp_gaussian_state(
    n_particles: int,
    temp_k: float,
    omega_phys_hz: Tuple[float, float, float],
    t_end_s: float,
    dt_s: float,
    r0_phys: float = None,
    C_phys: float = None,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Bereitet eine Simulation für eine Teilchengruppe im 
    thermischen Gleichgewicht im HO (Maxwell-Boltzmann) vor. für rb87 atome
    """
    mass_kg = 86.909 * 1.66054e-27 # kg 
    temp_ref_k = temp_k
    print(f"--- Skalierung basierend auf T = {temp_ref_k:.2e} K ---")
    omega_phys_rad_s = 2 * math.pi * torch.tensor(omega_phys_hz, dtype=precision, device=device)
    m0, E0 = mass_kg, kB * temp_ref_k
    omega_char_rad_s = omega_phys_rad_s[0]

    L0 = torch.sqrt(E0 / (m0 * omega_char_rad_s**2)) # Aus harmonischem Potenzial: E ~ m * w^2 * L^2
    T0 = 1.0 / omega_char_rad_s
    P0 = torch.sqrt(torch.tensor(m0 * E0, dtype=precision, device=device))

    print(f"Längenskala L0: {L0:.2e} m, Energieskala E0: {E0:.2e} J, Zeitskala T0: {T0:.2e} s")
    print(f"Initialisiere {n_particles} Teilchen bei T = {temp_k:.2e} K")
    
    sigma_q_phys_sq = (temp_k * kB) / (m0 * omega_phys_rad_s**2)
    sigma_q_phys = torch.sqrt(sigma_q_phys_sq)
    
    sigma_q_dimless_explicit = sigma_q_phys / L0
    
    q0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_q_dimless_explicit

    temp_ratio = temp_k / temp_ref_k
    sigma_p_dimless = math.sqrt(temp_ratio)
    
    p0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
    
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True)
    
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)
    
    omega_dimless = omega_phys_rad_s / omega_char_rad_s
    omega_matrix_dimless = torch.diag(omega_dimless)

    pair_params = {'r': 0.0, 'c': 0.0}
    if r0_phys is not None and C_phys is not None:
        pair_r_dimless = r0_phys / L0
        pair_c_dimless = C_phys / E0
        
        pair_params['r'] = float(pair_r_dimless)
        pair_params['c'] = float(pair_c_dimless)

        
    print("--- Initialisierung abgeschlossen ---\n")
    return {
        "t_values": t_values,
        "q0": q0_dimless,
        "p0": p0_dimless,
        "omega_matrix": omega_matrix_dimless.to(device),
        "pair_force_params": pair_params,
        "precision_type": precision,
        "device": device,
        "L0_m": L0,
        "T0_s": T0,
        "E0_J": E0
    }

def initialize_two_temp_gaussian_state(
    n_particles_groups: Tuple[int, int],
    temp_k_groups: Tuple[float, float],
    omega_phys_hz: Tuple[float, float, float],
    t_end_s: float,
    dt_s: float,
    r0_phys: float = None,
    C_phys: float = None,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Bereitet die Simulation vor 
    """
    mass_kg = 86.909 * 1.66054e-27 # kg 
    temp_ref_k = temp_k_groups[0]  ### nimm erste temperatur 
    print(f"--- Skalierung basierend auf Referenztemperatur T_ref = {temp_ref_k:.2e} K ---")
    omega_phys_rad_s = 2 * math.pi * torch.tensor(omega_phys_hz, dtype=precision, device=device)
    m0, E0 = mass_kg, kB * temp_ref_k
    omega_char_rad_s = omega_phys_rad_s[0]

    # --- Skalen berechnen ---
    L0 = torch.sqrt(E0 / (m0 * omega_char_rad_s**2))
    T0 = 1.0 / omega_char_rad_s
    P0 = torch.sqrt(torch.tensor(m0 * E0, dtype=precision, device=device))
    print(f"Längenskala L0: {L0:.2e} m, Energieskala E0: {E0:.2e} J, Zeitskala T0: {T0:.2e} s")
    
    q_list, p_list = [], []

    # --- Schleife für gruppen ---
    for n_particles, temp_k in zip(n_particles_groups, temp_k_groups):
        if n_particles == 0: continue
        print(f"Initialisiere Gruppe mit {n_particles} Teilchen bei T = {temp_k:.2e} K")
        
        # --- 1. q-Skalierung  ---
        # Normalverteilung mit standartabweichung 
        sigma_q_phys_sq = (temp_k * kB) / (m0 * omega_phys_rad_s**2)
        sigma_q_phys = torch.sqrt(sigma_q_phys_sq)
        
        # Skalierung dimless
        sigma_q_dimless_explicit = sigma_q_phys / L0
        
        q_group = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_q_dimless_explicit

        # --- 2. p-Skalierung  ---
        temp_ratio = temp_k / temp_ref_k
        sigma_p_dimless = math.sqrt(temp_ratio)
        p_group = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
        
        q_list.append(q_group)
        p_list.append(p_group)

    # --- zusammenfügen ---
    q0_dimless, p0_dimless = torch.cat(q_list, dim=0), torch.cat(p_list, dim=0)
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True)
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)
    
    # --- Korrektur für omega---
    omega_dimless = omega_phys_rad_s / omega_char_rad_s
    omega_matrix_dimless = torch.diag(omega_dimless)

    # --- Interaction parameter dimless machen ---
    pair_params = {'r': 0.0, 'c': 0.0}
    if r0_phys is not None and C_phys is not None:
        print("Skaliere Interaktionsparameter...")
        pair_r_dimless = r0_phys / L0
        pair_c_dimless = C_phys / E0
        
        pair_params['r'] = float(pair_r_dimless)
        pair_params['c'] = float(pair_c_dimless)
        
    print("--- Zwei-Temperatur-Initialisierung abgeschlossen ---\n")
    return {
        "t_values": t_values,
        "q0": q0_dimless,
        "p0": p0_dimless,
        "omega_matrix": omega_matrix_dimless.to(device),
        "pair_force_params": pair_params,
        "precision_type": precision,
        "device": device,
        "L0_m": L0,
        "T0_s": T0,
        "E0_J": E0
    }
