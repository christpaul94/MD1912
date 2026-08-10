"""
Module: InitialisationV2.py
Author: Paul Christ
Date: August 2026

Description:
This module provides functions to calculate dimensionless scaling factors 
and to initialize thermal states (positions and momenta) for N-body 
molecular dynamics simulations of trapped atoms.

It implements two distinct scaling paradigms:
1. Dynamic Scaling (Legacy): Dimensionless scales depend directly on the 
   trap frequencies. Ideal for isolated simulations to maximize float32 precision.
2. Fixed Scaling (Recommended): Dimensionless scales are derived from fixed 
   reference values (e.g., L0 = 1.0 µm, T_ref = 1.0 µK), ensuring robustness 
   and consistency across parameter sweeps and varying trap geometries.
"""

import math
import torch
from typing import Dict, Tuple

kB = 1.380649e-23  # J/K
MASS_RB87_KG = 86.909 * 1.66054e-27  # kg

# =====================================================================
# 1. DYNAMISCHE SKALIERUNG (Fallenabhängig)
# =====================================================================

def calculate_dimensionless_scales(
    temp_ref_k: float,
    omega_phys_hz: Tuple[float, float, float],
    mass_kg: float = MASS_RB87_KG,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Berechnet alle grundlegenden Umrechnungsfaktoren (Skalen) 
    für die dynamische, fallenabhängige Skalierung.
    """
    print(f"--- Skalierung basierend auf T_ref = {temp_ref_k:.2e} K ---")
    
    omega_phys_rad_s = 2 * math.pi * torch.tensor(omega_phys_hz, dtype=precision, device=device)
    omega_char_rad_s = omega_phys_rad_s[0] # Wir nutzen die x-Achse als Referenzfrequenz
    
    # Basis-Energieskala und Massenskala
    E0 = kB * temp_ref_k
    m0 = mass_kg
    
    # Abgeleitete Skalen (Länge, Zeit, Impuls)
    L0 = torch.sqrt(E0 / (m0 * omega_char_rad_s**2))
    T0 = 1.0 / omega_char_rad_s
    P0 = torch.sqrt(torch.tensor(m0 * E0, dtype=precision, device=device))
    
    print(f"Längenskala L0: {L0:.2e} m, Energieskala E0: {E0:.2e} J, Zeitskala T0: {T0:.2e} s")
    
    return {
        "m0_kg": m0,
        "E0_J": E0,
        "L0_m": L0.item(),  
        "T0_s": T0.item(),
        "P0_kg_m_s": P0.item(),
        "omega_phys_rad_s": omega_phys_rad_s,
        "omega_char_rad_s": omega_char_rad_s,
        "precision": precision,
        "device": device
    }

def initialize_dynamic_gaussian_state(
    n_particles: int,
    temp_k: float,
    scales: Dict,
    t_end_s: float,
    dt_s: float,
    r0_phys: float = None,
    C_phys: float = None
) -> Dict:
    """
    Bereitet den dimensionslosen Startzustand (q0, p0) mit dynamischen Skalen vor.
    """
    print(f"Initialisiere {n_particles} Teilchen bei T = {temp_k:.2e} K (dynamische Skalen)")
    
    precision = scales["precision"]
    device = scales["device"]
    L0 = scales["L0_m"]
    T0 = scales["T0_s"]
    E0 = scales["E0_J"]
    m0 = scales["m0_kg"]
    omega_phys_rad_s = scales["omega_phys_rad_s"]
    
    # --- Ortsverteilung (q0) ---
    sigma_q_phys_sq = (temp_k * kB) / (m0 * omega_phys_rad_s**2)
    sigma_q_phys = torch.sqrt(sigma_q_phys_sq)
    sigma_q_dimless = sigma_q_phys / L0
    
    q0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_q_dimless

    # --- Impulsverteilung (p0) ---
    temp_ref_k = E0 / kB
    temp_ratio = temp_k / temp_ref_k
    sigma_p_dimless = math.sqrt(temp_ratio)
    
    p0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True) 
    
    # --- Zeitachse und Fallenmatrix ---
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)
    
    omega_dimless = omega_phys_rad_s / scales["omega_char_rad_s"]
    omega_matrix_dimless = torch.diag(omega_dimless)

    # --- Wechselwirkungsparameter ---
    pair_params = {'sigma': 0.0, 'V0': 0.0}
    if r0_phys is not None and C_phys is not None:
        pair_params['sigma'] = float(r0_phys / L0)
        pair_params['V0'] = float(C_phys / E0)

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

# =====================================================================
# 2. FESTE SKALIERUNG (Fallenunabhängig)
# =====================================================================

def calculate_fixed_scales(
    L0_um: float = 1.0,
    T_ref_k: float = 1.0e-6,
    mass_kg: float = MASS_RB87_KG,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Berechnet feste, fallenunabhängige Skalen für eine robuste Vergleichbarkeit.
    """
    L0 = L0_um * 1e-6
    E0 = kB * T_ref_k
    m0 = mass_kg
    T0 = L0 * math.sqrt(m0 / E0)

    print(f"→ Konsistente Skalen: L0 = {L0_um} µm | E0 = {E0:.2e} J | T0 = {T0*1000:.2f} ms")
    return {"L0_m": L0, "E0_J": E0, "T0_s": T0, "m0_kg": m0, "precision": precision, "device": device}

def initialize_fixed_gaussian_state(
    n_particles: int,
    temp_k: float,
    scales: Dict,
    t_end_s: float,
    dt_s: float,
    omega_phys_rad_s: torch.Tensor,
    r0_phys: float = None,
    C_phys: float = None
) -> Dict:
    """
    Initialisiert den thermischen Zustand basierend auf den festen Skalen.
    """
    precision = scales["precision"]
    device = scales["device"]
    L0 = scales["L0_m"]
    T0 = scales["T0_s"]
    E0 = scales["E0_J"]
    m0 = scales["m0_kg"]

    print(f"Initialisiere {n_particles} Teilchen bei T = {temp_k:.2e} K (feste Skalen)")

    # Positionsverteilung
    sigma_q_phys_sq = (temp_k * kB) / (m0 * omega_phys_rad_s**2)
    sigma_q_phys = torch.sqrt(sigma_q_phys_sq)
    sigma_q_dimless = sigma_q_phys / L0
    q0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_q_dimless

    # Impulsverteilung
    temp_ref_k = E0 / kB
    temp_ratio = temp_k / temp_ref_k
    sigma_p_dimless = math.sqrt(temp_ratio)
    p0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True)

    # Zeitachse
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)

    # Trap-Matrix (Multiplikation mit T0 statt Division)
    omega_dimless = omega_phys_rad_s * T0
    omega_matrix_dimless = torch.diag(omega_dimless)

    # Wechselwirkung
    pair_params = {'sigma': 0.0, 'V0': 0.0}
    if r0_phys is not None and C_phys is not None:
        pair_params['sigma'] = float(r0_phys / L0)
        pair_params['V0'] = float(C_phys / E0)

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
