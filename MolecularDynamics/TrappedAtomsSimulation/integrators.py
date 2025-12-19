import torch
from typing import Dict, Tuple, Callable
import time
import torch
from typing import Dict, Tuple, Callable
import time



### Harmonic oscillator Teil 

def harmonic_fp(
    q: torch.Tensor,
    omega_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega_squared = omega_matrix @ omega_matrix.T
    forces = - q @ omega_squared
    potential_per_particle = 0.5 * torch.einsum('ni,ij,nj->n', q, omega_squared, q)
    total_potential = potential_per_particle.sum()
    return forces, total_potential, potential_per_particle


def no_force_fp(q: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fallenfunktion, die keine Kraft und kein Potential zurückgibt.
    """
    forces = torch.zeros_like(q)
    total_potential = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    potential_per_particle = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    return forces, total_potential, potential_per_particle

def no_pair_force_fp(q: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Funktion für die Paar-Wechselwirkung (pair_force_func), 
    die keine Kraft und kein Potential zurückgibt.
    """
    forces = torch.zeros_like(q)
    total_potential = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    return forces, total_potential


def solve_harmonic_analytical(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    mass: float = 1.0,  
    **kwargs 
):

    device, dtype = q0.device, q0.dtype
    nT, n_particles = t_values.shape[0], q0.shape[0]

    q_out = torch.empty((nT, n_particles, 3), device=device, dtype=dtype)
    p_out = torch.empty((nT, n_particles, 3), device=device, dtype=dtype)
    kinetic_energy_out = torch.empty((nT,), device=device, dtype=dtype)
    potential_harmonic_out = torch.empty((nT,), device=device, dtype=dtype)
    potential_pair_out = torch.zeros((nT,), device=device, dtype=dtype) 
    
 
    omegas = torch.diagonal(omega_matrix) 
    
 
    for i, t in enumerate(t_values):

  
        c_t = torch.cos(omegas * t) 
        s_t = torch.sin(omegas * t)
        
 
        q = q0 * c_t + (p0 / (mass * omegas)) * s_t
        p = - (mass * omegas) * q0 * s_t + p0 * c_t

        q_out[i] = q
        p_out[i] = p

        # Energies
        kinetic_energy_out[i] = 0.5 * torch.sum(p**2) / mass # Falls mass=1, ist /1 egal aber korrekt
        
 
        pot_per_particle = 0.5 * mass * torch.sum( (omegas**2) * (q**2), dim=1)
        
        potential_harmonic_out[i] = pot_per_particle.sum()

    return {
        "times": t_values,
        "positions": q_out,
        "momenta": p_out,
        "kinetic_energy": kinetic_energy_out,
        "potential_energy_harmonic": potential_harmonic_out,
        "potential_energy_pair": potential_pair_out
    }


def harmonic_fp(
    q: torch.Tensor,
    omega_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega_squared = omega_matrix @ omega_matrix.T
    forces = - q @ omega_squared
    potential_per_particle = 0.5 * torch.einsum('ni,ij,nj->n', q, omega_squared, q)
    total_potential = potential_per_particle.sum()
    return forces, total_potential, potential_per_particle

def run_verlet_simulation_HO(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    pair_force_func: Callable,
    pair_force_params: Dict,
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Führt eine Simulation mit dem Velocity Verlet-Algorithmus durch.

    """
    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_harmonic_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Initialisierung ---
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        f_h_current, pot_h, _ = harmonic_fp(q_current, omega_matrix)
        f_p_current, pot_p = pair_force_func(q_current, **pair_force_params)
        a_current = f_h_current + f_p_current
        potential_harmonic_out[0], potential_pair_out[0] = pot_h, pot_p

        p_half = torch.empty_like(p_current)
        q_next = torch.empty_like(q_current)
        a_next = torch.empty_like(a_current)
        p_next = torch.empty_like(p_current)

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps

            # --- INNERE SCHLEIFE ---
            for _ in range(substeps):
                p_half = p_current + 0.5 * dt * a_current
                q_next = q_current + dt * p_half

                f_h_next, pot_h_next, _ = harmonic_fp(q_next, omega_matrix)
                f_p_next, pot_p_next = pair_force_func(q_next, **pair_force_params)
                a_next = f_h_next + f_p_next

                p_next = p_half + 0.5 * dt * a_next

                # Verwende In-Place-Swapping für Puffer
                q_current, q_next = q_next, q_current
                p_current, p_next = p_next, p_current
                a_current, a_next = a_next, a_current
            # --- ENDE INNERE SCHLEIFE ---

            # Zustand am Ende in die Output-Arrays schreiben
            q_out[i], p_out[i] = q_current, p_current
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_harmonic_out[i] = pot_h_next
            potential_pair_out[i] = pot_p_next

            loop_end_time = time.perf_counter()
            eta_seconds = int((loop_end_time - loop_start_time) * (num_save_points - 1 - i))
            print(f"\rIntegration {100 * (i + 1) / num_save_points:.0f}% "
                  f"| Time: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)



        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_harmonic": potential_harmonic_out,
            "potential_energy_pair": potential_pair_out
        }



def run_DKDsplitting_simulation_HO(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    pair_force_func: Callable,
    pair_force_params: Dict,
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Simulation mit dem Strang Splitting (Drift-Kick-Drift)
    
    H = T(p) + V(q)

    """
    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        # Output-Arrays
        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_harmonic_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Initialisierung ---
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        
        # Berechne Potentiale für die Speicherung
      
        _, pot_h, _ = harmonic_fp(q_current, omega_matrix)
        _, pot_p = pair_force_func(q_current, **pair_force_params)
        potential_harmonic_out[0], potential_pair_out[0] = pot_h, pot_p


        q_half = torch.empty_like(q_current)

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps
            dt_half = 0.5 * dt # Halber Zeitschritt

            # --- INNERE SCHLEIFE (Drift-Kick-Drift) ---
            for _ in range(substeps):
                
                # 1. DRIFT (dt/2): p ist konstant
                # q_half = q_current + p_current * dt_half
                torch.add(q_current, p_current, alpha=dt_half, out=q_half)
                
                # 2. KICK (dt): q ist konstant
                
                f_h_half, pot_h_half, _ = harmonic_fp(q_half, omega_matrix)
                f_p_half, pot_p_half = pair_force_func(q_half, **pair_force_params)
                
                # p_current = p_current + (f_h_half + f_p_half) * dt
                p_current.add_(f_h_half, alpha=dt)
                p_current.add_(f_p_half, alpha=dt)
                
                # 3. DRIFT (dt/2): p ist konstant
                # q_current = q_half + p_current * dt_half
                torch.add(q_half, p_current, alpha=dt_half, out=q_current)

            # --- ENDE INNERE SCHLEIFE ---

            # Zustand am Ende in die Output-Arrays schreiben
            q_out[i], p_out[i] = q_current, p_current
            
            # Energien berechnen 
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_harmonic_out[i] = pot_h_half
            potential_pair_out[i] = pot_p_half

            loop_end_time = time.perf_counter()
            eta_seconds = int((loop_end_time - loop_start_time) * (num_save_points - 1 - i))
            print(f"\rIntegration {100 * (i + 1) / num_save_points:.0f}% "
                  f"| Time: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)

        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_harmonic": potential_harmonic_out,
            "potential_energy_pair": potential_pair_out
        }

### Dipole Trap Teil 



def run_verlet_simulation_general(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    trap_force_func: Callable,   # calculate_crossed_beam_dipole_potential
    trap_force_params: Dict,    # Parameter für die Falle
    pair_force_func: Callable,    # pair_keops_fp
    pair_force_params: Dict,    # Parameter für die WW
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    silent: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """simulation mit Velocity Verlet """
    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_trap_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Initialisierung ---
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        f_trap_current, pot_trap, _ = trap_force_func(q_current, **trap_force_params)
        f_pair_current, pot_pair = pair_force_func(q_current, **pair_force_params)
        a_current = f_trap_current + f_pair_current
        potential_trap_out[0], potential_pair_out[0] = pot_trap, pot_pair

        p_half = torch.empty_like(p_current)
        q_next = torch.empty_like(q_current)
        a_next = torch.empty_like(a_current)
        p_next = torch.empty_like(p_current)

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps

            for _ in range(substeps):
                p_half = p_current + 0.5 * dt * a_current
                q_next = q_current + dt * p_half

                f_trap_next, pot_trap_next, _ = trap_force_func(q_next, **trap_force_params)
                f_pair_next, pot_pair_next = pair_force_func(q_next, **pair_force_params)
                a_next = f_trap_next + f_pair_next
                p_next = p_half + 0.5 * dt * a_next

                q_current, q_next = q_next, q_current
                p_current, p_next = p_next, p_current
                a_current, a_next = a_next, a_current

            q_out[i], p_out[i] = q_current, p_current
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_trap_out[i] = pot_trap_next
            potential_pair_out[i] = pot_pair_next

            loop_end_time = time.perf_counter()
            eta_seconds = int((loop_end_time - loop_start_time) * (num_save_points - 1 - i))
            print(f"\rIntegration {100 * (i + 1) / num_save_points:.0f}% "
                  f"| ETA: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)
        
        print("\nIntegration abgeschlossen.")

        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_trap": potential_trap_out,
            "potential_energy_pair": potential_pair_out
        }



import torch
import time
from typing import Callable, Dict, Union, List

def run_verlet_simulation_dynamic(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    trap_force_func: Callable,
    # Änderung: Akzeptiert jetzt Dict (statisch) oder List[Dict] (dynamisch)
    trap_force_params: Union[Dict, List[Dict]], 
    pair_force_func: Callable,
    pair_force_params: Dict,
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    silent: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:

    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        # Prüfung: Sind die Parameter statisch oder dynamisch?
        is_dynamic_trap = isinstance(trap_force_params, list)
        
        if is_dynamic_trap:
            if len(trap_force_params) != num_save_points:
                raise ValueError(f"Länge der trap_force_params Liste ({len(trap_force_params)}) "
                                 f"muss mit t_values ({num_save_points}) übereinstimmen.")
            # Start-Parameter für t=0
            current_trap_params = trap_force_params[0]
        else:
            # Statische Parameter (wie bisher)
            current_trap_params = trap_force_params

        # --- Allocations (wie zuvor) ---
        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_trap_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q = q0.to(device, precision_type).clone()
        p = p0.to(device, precision_type).clone()

        # --- 1. Initiale Kraftberechnung (t=0) ---
        f_trap, pot_trap, _ = trap_force_func(q, **current_trap_params)
        f_pair, pot_pair = pair_force_func(q, **pair_force_params)
        a = f_trap + f_pair 

        # Speichern t=0
        q_out[0] = q
        p_out[0] = p
        kinetic_energy_out[0] = 0.5 * torch.sum(p**2)
        potential_trap_out[0] = pot_trap
        potential_pair_out[0] = pot_pair

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps
            dt_half = 0.5 * dt

            # --- VORBEREITUNG NEUER ZEITSCHRITT ---
            if is_dynamic_trap:
                # 1. Parameter für den aktuellen Schritt holen
                # Wir nehmen an: Parameter ändern sich JETZT und gelten für das Intervall [t_i-1, t_i]
                # Alternativ: trap_force_params[i-1]. Das ist Definitionssache.
                # Meistens will man die Parameter, die zum Zielzeitpunkt t[i] führen.
                current_trap_params = trap_force_params[i]

                # 2. KRITISCH: Kräfte neu berechnen!
                # Da sich das Potential V geändert hat, ist die Kraft F(q_current)
                # aus dem letzten Loop-Durchlauf nicht mehr korrekt.
                # Wir müssen die Kraft am aktuellen Ort q mit den NEUEN Parametern updaten.
                
                f_trap, pot_trap, _ = trap_force_func(q, **current_trap_params)
                # f_pair ändert sich nicht, aber a muss neu berechnet werden
                # (f_pair ist noch vom Ende des letzten Schritts im Speicher, das ist okay)
                a = f_trap + f_pair 
            
            # --- Substeps (Velocity Verlet) ---
            # Hier verwenden wir `current_trap_params` konstant für die Substeps
            for _ in range(substeps):
                p.add_(a, alpha=dt_half) 
                q.add_(p, alpha=dt)

                f_trap, pot_trap, _ = trap_force_func(q, **current_trap_params)
                f_pair, pot_pair = pair_force_func(q, **pair_force_params)
                
                a = f_trap + f_pair
                p.add_(a, alpha=dt_half)

            # --- Speichern ---
            q_out[i].copy_(q)
            p_out[i].copy_(p)
            kinetic_energy_out[i] = 0.5 * torch.sum(p**2)
            potential_trap_out[i] = pot_trap
            potential_pair_out[i] = pot_pair

            if not silent:
                # ... (Logging Code wie gehabt) ...
                pass # Platzhalter für Übersichtlichkeit
        
        print("\nIntegration abgeschlossen.")

        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_trap": potential_trap_out,
            "potential_energy_pair": potential_pair_out
        }

