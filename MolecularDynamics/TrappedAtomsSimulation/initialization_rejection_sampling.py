from typing import Dict, Tuple, Callable

import torch
import math
from typing import Dict, Tuple, Callable

# Boltzmann-Konstante
kB = 1.380649e-23

 
def prepare_one_temp_simulation_rejection(
    n_particles: int,
    temp_k: float,
    omega_phys_hz: Tuple[float, float, float],  

    # Parameter für  Falle
    trap_force_func: Callable,
    trap_params: Dict,

    t_end_s: float,
    dt_s: float,
    r0_phys: float = None,
    C_phys: float = None,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Rejection Sampling i 
    """
    # macht das gleiche wie kslaierung bei der gaußverteilten Initialisierung,
    # aber initialisiert die Positionen via Rejection Sampling. vielleicht kann man von gauss intitalisierung was wiederverwenden

    # --- 1. Skalierung (basierend auf der einen Gruppe) ---
    mass_kg = 86.909 * 1.66054e-27
    temp_ref_k = temp_k  # T_ref ist einfach die Temperatur der Gruppe
    print(f"--- Skalierung basierend auf T_ref = {temp_ref_k:.2e} K ---")

    omega_phys_rad_s = 2 * math.pi * torch.tensor(omega_phys_hz, dtype=precision, device=device)
    m0, E0 = mass_kg, kB * temp_ref_k
    omega_char_rad_s = omega_phys_rad_s[0]
    L0 = torch.sqrt(E0 / (m0 * omega_char_rad_s**2))
    T0 = 1.0 / omega_char_rad_s
    print(f"Längenskala L0: {L0:.2e} m, Energieskala E0: {E0:.2e} J...")

    # --- 2. Bounding Box für Sampling ---
    # Die Bounding Box basiert ebenfalls auf dieser einen Temperatur
    sigma_q_phys_sq = (temp_k * kB) / (m0 * omega_phys_rad_s**2)
    sigmas_dimless = torch.sqrt(sigma_q_phys_sq) / L0
    print(f"Verwende Bounding Box basierend auf T = {temp_k:.2e} K.")

    # --- 3. Initialisierung der Gruppe   ---
    print(f"Initialisiere Gruppe (N={n_particles}, T={temp_k:.2e} K)...")

    # Das Temperatur-Verhältnis ist T_k / T_ref = T_k / T_k = 1.0 da es nur eine Gruppe gibt
    temp_ratio = 1.0

    # --- Positionen (Rejection Sampling) ---
    q0_dimless = generate_positions_rejection_sampling(
        n_particles_target=n_particles,
        sigmas_dimless=sigmas_dimless, # Verwende die berechnete Box
        trap_force_func=trap_force_func,
        trap_params=trap_params,
        temp_ratio=temp_ratio, # Ist 1.0 da es nur eine Gruppe gibt
        device=device,
        precision=precision
    )

    # --- Impulse (Gauß-Verteilung) ---
    # sigma_p_dimless = sqrt(temp_ratio) = sqrt(1.0) = 1.0
    sigma_p_dimless = 1.0
    p0_dimless = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
    
    # Gesamtimpuls des Systems zentrieren
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True)

    # --- 4. Rest (wie zuvor) ---
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)
    omega_dimless = omega_phys_rad_s / omega_char_rad_s
    omega_matrix_dimless = torch.diag(omega_dimless)

    pair_params = {'r': 0.0, 'c': 0.0}
    if r0_phys is not None and C_phys is not None:
        pair_r_dimless = r0_phys / L0
        pair_c_dimless = C_phys / E0
        pair_params['r'] = float(pair_r_dimless)
        pair_params['c'] = float(pair_c_dimless)
        print(f"Dimensionslose Paar-Parameter: r={pair_params['r']:.5f}, c={pair_params['c']:.3e}")

    print("--- Ein-Temperatur-Initialisierung abgeschlossen ---\n")
    return {
        "t_values": t_values,
        "q0": q0_dimless,
        "p0": p0_dimless,
        "omega_matrix_dimless": omega_matrix_dimless,
        "pair_force_params": pair_params,
        "precision_type": precision,
        "device": device,
        "L0_m": L0,
        "T0_s": T0,
        "E0_J": E0
    }


def generate_positions_rejection_sampling(
    n_particles_target: int,
    sigmas_dimless: torch.Tensor, # Bounding-Box-Größe
    trap_force_func: Callable,
    trap_params: Dict,
    temp_ratio: float,           # T / T_ref
    device: torch.device,
    precision: torch.dtype,
    batch_size: int = 10000,
    box_scale: float = 15.0
) -> torch.Tensor:
    """
    Generiert Teilchenpositionen mittels Rejection Sampling.
    """
    print(f"  -> Starte Rejection Sampling für {n_particles_target} Teilchen "
          f"(T/T_ref = {temp_ratio:.2f})...")
    accepted_positions = []
    n_accepted = 0
    n_total_tried = 0

    # 1.  U_min berechenen
    q_origin = torch.zeros((1, 3), device=device, dtype=precision)
    _force, U_min_tensor, _pot_p = trap_force_func(q_origin, **trap_params)
    U_min = U_min_tensor.item() 

    # 2. Bounding Box
    box_lims_low = -box_scale * sigmas_dimless
    box_lims_high = box_scale * sigmas_dimless

    while n_accepted < n_particles_target:
        n_needed = n_particles_target - n_accepted

        # 3. Proposal verteilung
        q_proposals = torch.rand(batch_size, 3, device=device, dtype=precision) \
                      * (box_lims_high - box_lims_low) + box_lims_low
        
        # Erzeugt gleichverteilte Zufallspunkte im Intervall [box_lims_low, box_lims_high]:
        # 1. torch.rand(...): Liefert Werte im Bereich [0, 1)
        # 2. * (... - ...):   Skaliert  den Bereich auf die Breite der Box
        # 3. + box_lims_low:  Verschiebt den Bereich zum korrekten Startpunkt
        n_total_tried += batch_size

        # 4.  Akzeptanz-Wahrscheinlichkeit
        _f, _pot_total_batch, pot_per_particle = trap_force_func(q_proposals, **trap_params)

        delta_U = pot_per_particle - U_min

        # P_accept = exp(-DeltaU / (k_B T / E_0)) = exp(-DeltaU / temp_ratio)
        p_accept = torch.exp(-delta_U / temp_ratio)
 
        # 5. Accept/Reject
        u = torch.rand(batch_size, device=device, dtype=precision)
        mask_accepted = (u < p_accept)

        new_positions = q_proposals[mask_accepted]

        if new_positions.shape[0] > 0:     ## falls kein neues Teilchen akzeptiert wurde, nichts tun 
            if n_accepted + new_positions.shape[0] > n_particles_target: # zu viele akzeptiert 
                new_positions = new_positions[:n_needed]
            accepted_positions.append(new_positions)   # akzeptierte Positionen speichern
            n_accepted += new_positions.shape[0] # Anzahl akzeptierter Teilchen aktualisieren

    print(f"  -> Sampling beendet. Akzeptanzrate: {n_particles_target / n_total_tried * 100:.2f}%")
    return torch.cat(accepted_positions, dim=0)

def prepare_two_temp_simulation_rejection(
    n_particles_groups: Tuple[int, int],
    temp_k_groups: Tuple[float, float],
    omega_phys_hz: Tuple[float, float, float], # Für Bounding Box

    # Parameter Falle
    trap_force_func: Callable,
    trap_params: Dict,

    t_end_s: float,
    dt_s: float,
    r0_phys: float = None,
    C_phys: float = None,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Bereitet eine Zwei-Temperaturen-Simulation vor, indem Gruppen
     mittels Rejection Sampling initialisiert werden.
    """

    # --- 1. Skalierung (basierend auf Gruppe 1) ---
    mass_kg = 86.909 * 1.66054e-27
    temp_ref_k = temp_k_groups[0]
    print(f"--- Skalierung basierend auf T_ref = {temp_ref_k:.2e} K (Gruppe 1) ---")

    omega_phys_rad_s = 2 * math.pi * torch.tensor(omega_phys_hz, dtype=precision, device=device)
    m0, E0 = mass_kg, kB * temp_ref_k
    omega_char_rad_s = omega_phys_rad_s[0]
    L0 = torch.sqrt(E0 / (m0 * omega_char_rad_s**2))
    T0 = 1.0 / omega_char_rad_s
    print(f"Längenskala L0: {L0:.2e} m, Energieskala E0: {E0:.2e} J...")

    # --- 2. Bounding Box für Sampling ---
# box muss auch für die heißere gruppe groß genug sein
    temp_max_k = max(temp_k_groups)
    sigma_q_phys_sq_max = (temp_max_k * kB) / (m0 * omega_phys_rad_s**2)
    sigmas_dimless_max = torch.sqrt(sigma_q_phys_sq_max) / L0
    print(f"Verwende Bounding Box basierend auf T_max = {temp_max_k:.2e} K.")

    q_list = []
    p_list = []

    # --- 3. Schleife über die beiden Gruppen ---
    for n_particles, temp_k in zip(n_particles_groups, temp_k_groups):
        if n_particles == 0:
            continue

        print(f"Initialisiere Gruppe (N={n_particles}, T={temp_k:.2e} K)...")

        # Berechne das Temperatur-Verhältnis für diese Gruppe
        temp_ratio = temp_k / temp_ref_k

        # --- Positionen (Rejection Sampling) ---
        q_group = generate_positions_rejection_sampling(
            n_particles_target=n_particles,
            sigmas_dimless=sigmas_dimless_max, # Verwende die große Box
            trap_force_func=trap_force_func,
            trap_params=trap_params,
            temp_ratio=temp_ratio, # Wichtig!
            device=device,
            precision=precision
        )

        # --- Impulse (Gauß-Verteilung) ---
        sigma_p_dimless = math.sqrt(temp_ratio)
        p_group = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
        p_group -= torch.mean(p_group, dim=0, keepdim=True) # Impuls der Gruppe zentrieren

        q_list.append(q_group)
        p_list.append(p_group)

    # --- 4. Kombinieren & Abschluss ---
    q0_dimless = torch.cat(q_list, dim=0)
    p0_dimless = torch.cat(p_list, dim=0)

    # Gesamtimpuls des Systems zentrieren
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True)

    # 
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)
    omega_dimless = omega_phys_rad_s / omega_char_rad_s
    omega_matrix_dimless = torch.diag(omega_dimless)

    pair_params = {'r': 0.0, 'c': 0.0}
    if r0_phys is not None and C_phys is not None:
        pair_r_dimless = r0_phys / L0
        pair_c_dimless = C_phys / E0
        pair_params['r'] = float(pair_r_dimless)
        pair_params['c'] = float(pair_c_dimless)
        print(f"Dimensionslose Paar-Parameter: r={pair_params['r']:.5f}, c={pair_params['c']:.3e}")

    print("--- Zwei-Temperaturen-Initialisierung abgeschlossen ---\n")
    return {
        "t_values": t_values,
        "q0": q0_dimless,
        "p0": p0_dimless,
        "omega_matrix_dimless": omega_matrix_dimless, # Für harmonische Näherung
        "pair_force_params": pair_params,
        "precision_type": precision,
        "device": device,
        "L0_m": L0,
        "T0_s": T0,
        "E0_J": E0
    }
