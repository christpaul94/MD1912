import torch
import math
from typing import Dict, Tuple

import torch
import math
from typing import Tuple


kB = 1.380649e-23  # J/K
c_light = 299_792_458.0 # m/s
pi = math.pi

### ANGEPASSTES MODELL (Extended Model D1/D2)

#@torch.compile()
def calculate_single_beam_X_AXIS(
    positions: torch.Tensor,
    P: float, w0: float, s0: float,  
    # --- Parameter für Extended Model ---
    omega_L: float,    # Laser-Kreisfrequenz (rad/s)
    omega_0_D1: float, # D1 Atom-Kreisfrequenz (rad/s)
    Gamma_D1: float,   # D1 Linienbreite (rad/s)
    omega_0_D2: float, # D2 Atom-Kreisfrequenz (rad/s)
    Gamma_D2: float,   # D2 Linienbreite (rad/s)
    # --- Skalierung ---
    L0: float,      
    E0: float       
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Berechnung für einen Gauß-Strahl, der entlang der X-ACHSE propagiert.
    Verwendet das "Extended Model" (D1- und D2-Linien).
    
    """
    with torch.no_grad():
        # --- 1. Skalierung & SI-Einheiten ---
        
        hbar = 1.054571817e-34
        c_light = 299_792_458.0
        
        r = positions * L0 # Verwende L0

        # --- 2. Geometrie (Hartkodiert für X-Achse) ---
        s = r[:, 0]  # s = x
        rho2 = r[:, 1].pow(2) + r[:, 2].pow(2) # rho^2 = y^2 + z^2

        # --- 3. Intensität & Potential ---
        w_s = w0 * torch.sqrt(1 + (s / s0).pow(2))
        intensity = (2 * P) / (math.pi * w_s.pow(2)) * torch.exp(-2 * rho2 / w_s.pow(2))
        
        # --- Extended Model Potentialberechnung (aus calculate_U0 übernommen) ---
        
        # Term für die D1-Linie
        term_D1 = (Gamma_D1 / omega_0_D1**3) * \
                    (1 / (omega_0_D1 - omega_L) + 1 / (omega_0_D1 + omega_L))
            
        # Term für die D2-Linie
        term_D2 = (2 * Gamma_D2 / omega_0_D2**3) * \
                    (1 / (omega_0_D2 - omega_L) + 1 / (omega_0_D2 + omega_L))

        # vorfaktor für das Potential U(r) = prefactor * I(r)
 
        potential_prefactor = -(math.pi * c_light**2) / 2 * (term_D1 + term_D2)
        
        potential_SI = potential_prefactor * intensity
 
        
        potential_dimless = potential_SI / E0 # Verwende E0
        total_potential = torch.sum(potential_dimless)

        # --- 4. Kraftberechnung ---
        dw_ds = w0 * (s / s0**2) / torch.sqrt(1 + (s / s0).pow(2))
        dU_ds = potential_SI * (4 * rho2 / w_s.pow(3) - 2 / w_s) * dw_ds
        dU_drho2 = -2 * potential_SI / w_s.pow(2)

        grads_U = torch.empty_like(r)
        
        # ∇U = (∂U/∂s) * ∇s + (∂U/∂ρ²) * ∇(ρ²)
        # ∇s = (1, 0, 0)
        # ∇(rho^2) = (0, 2y, 2z)
        
        grads_U[:, 0] = dU_ds              # dU/dx = dU/ds * 1
        grads_U[:, 1] = dU_drho2 * 2 * r[:, 1] # dU/dy = dU/d(rho^2) * 2y
        grads_U[:, 2] = dU_drho2 * 2 * r[:, 2] # dU/dz = dU/d(rho^2) * 2z

        
        forces_dimless = -grads_U * L0 / E0 # Verwende L0 und E0
        
        return forces_dimless, total_potential, potential_dimless


import torch
import math
from typing import Tuple

#@torch.compile()
def calculate_single_beam_Y_AXIS(
    positions: torch.Tensor,
    P: float, w0: float, s0: float,  
    # --- Parameter für Extended Model ---
    omega_L: float,    # Laser-Kreisfrequenz (rad/s)
    omega_0_D1: float, # D1 Atom-Kreisfrequenz (rad/s)
    Gamma_D1: float,   # D1 Linienbreite (rad/s)
    omega_0_D2: float, # D2 Atom-Kreisfrequenz (rad/s)
    Gamma_D2: float,   # D2 Linienbreite (rad/s)
    # --- Skalierungs-Parameter ---
    L0: float,  # Parameter (statt a_ho)
    E0: float   # Parameter (statt E_ho)
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Berechnung für einen Gauß-Strahl, der entlang der Y-ACHSE propagiert.
    Verwendet das "Extended Model" (D1- und D2-Linien).
    """
    with torch.no_grad():
        # --- 1. Skalierung & SI-Einheiten ---
        hbar = 1.054571817e-34
        c_light = 299_792_458.0
        
        r = positions * L0 # Verwende L0

        # --- 2. Geometrie ---
        s = r[:, 1]  # s = y
        rho2 = r[:, 0].pow(2) + r[:, 2].pow(2) # rho^2 = x^2 + z^2

        # --- 3. Intensität & Potential ---
        w_s = w0 * torch.sqrt(1 + (s / s0).pow(2))
        intensity = (2 * P) / (math.pi * w_s.pow(2)) * torch.exp(-2 * rho2 / w_s.pow(2))
        
        # --- Extended Model Potentialberechnung  --
        
        # Term für die D1-Linie
        term_D1 = (Gamma_D1 / omega_0_D1**3) * \
                    (1 / (omega_0_D1 - omega_L) + 1 / (omega_0_D1 + omega_L))
            
        # Term für die D2-Linie
        term_D2 = (2 * Gamma_D2 / omega_0_D2**3) * \
                    (1 / (omega_0_D2 - omega_L) + 1 / (omega_0_D2 + omega_L))

        # Pvorfactorfaktor für das Potential U(r) = prefactor * I(r)
    
        potential_prefactor = -(math.pi * c_light**2) / 2 * (term_D1 + term_D2)
        
        potential_SI = potential_prefactor * intensity
    
        
        potential_dimless = potential_SI / E0 
        total_potential = torch.sum(potential_dimless)

        # --- 4. Kraftberechnung ---
        dw_ds = w0 * (s / s0**2) / torch.sqrt(1 + (s / s0).pow(2))
        dU_ds = potential_SI * (4 * rho2 / w_s.pow(3) - 2 / w_s) * dw_ds
        dU_drho2 = -2 * potential_SI / w_s.pow(2)

        grads_U = torch.empty_like(r)
        
        # ∇U = (∂U/∂s) * ∇s + (∂U/∂ρ²) * ∇(ρ²)
        # ∇s = (0, 1, 0)
        # ∇(rho^2) = (2x, 0, 2z)
        
        grads_U[:, 0] = dU_drho2 * 2 * r[:, 0] # dU/dx = dU/d(rho^2) * 2x
        grads_U[:, 1] = dU_ds              # dU/dy = dU/ds * 1
        grads_U[:, 2] = dU_drho2 * 2 * r[:, 2] # dU/dz = dU/d(rho^2) * 2z

        # --- 5. Rückgabe ---
        forces_dimless = -grads_U * L0 / E0 # Verwende L0 und E0
        
        return forces_dimless, total_potential, potential_dimless

import torch
import math
from typing import Tuple

#@torch.compile()
def calculate_crossed_beam_dipole_potential(
    positions: torch.Tensor,
    P_x: float,        # Leistung des Strahls in x-Richtung
    P_y: float,        # Leistung des Strahls in y-Richtung
    w0_x: float,       # Strahl Taille für x-Strahl
    w0_y: float,       # Strahl Taille für y-Strahl
    s0_x: float,       # Rayleigh-Länge für x-Strahl
    s0_y: float,       # Rayleigh-Länge für y-Strahl
    # --- Parameter für Extended Model ---
    omega_L: float,    # Laser-Kreisfrequenz (rad/s)
    omega_0_D1: float, # D1 Atom-Kreisfrequenz (rad/s)
    Gamma_D1: float,   # D1 Linienbreite (rad/s)
    omega_0_D2: float, # D2 Atom-Kreisfrequenz (rad/s)
    Gamma_D2: float,   # D2 Linienbreite (rad/s)
    # --- Skalierungs-Parameter ---
    L0: float,         # NEU: Als Parameter übergeben
    E0: float,         # NEU: Als Parameter übergeben
    **kwargs           # Nimmt alle weiteren Argumente auf
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Berechnet das Potential und die Kräfte für eine gekreuzte Dipolfalle
    (Verwendet das Extended Model)
    """
 

    # --- 2. Berechnung für den Strahl in x-Richtung ---
    forces_x, pot_total_x, pot_particles_x = calculate_single_beam_X_AXIS(
        positions=positions,  
        P=P_x, w0=w0_x, s0=s0_x,  
        omega_L=omega_L,
        omega_0_D1=omega_0_D1, Gamma_D1=Gamma_D1,
        omega_0_D2=omega_0_D2, Gamma_D2=Gamma_D2,
        L0=L0, E0=E0   
    )

    # --- 3. Berechnung für den Strahl in y-Richtung ---
    forces_y, pot_total_y, pot_particles_y = calculate_single_beam_Y_AXIS(
        positions=positions,  
        P=P_y, w0=w0_y, s0=s0_y,  
        omega_L=omega_L,
        omega_0_D1=omega_0_D1, Gamma_D1=Gamma_D1,
        omega_0_D2=omega_0_D2, Gamma_D2=Gamma_D2,
        L0=L0, E0=E0   
    )

 
    total_forces = forces_x + forces_y
    total_potential = pot_total_x + pot_total_y
    total_potential_per_particle = pot_particles_x + pot_particles_y

    return total_forces, total_potential, total_potential_per_particle

 


def calculate_U0(
    P: float,        # Laserleistung (W)
    w0: float,       # Strahl-Taille (m)
    omega_L: float,    # Laser-Kreisfrequenz (rad/s)
    omega_0_D1: float, # D1 Atom-Kreisfrequenz (rad/s)
    Gamma_D1: float,   # D1 Linienbreite (rad/s)
    omega_0_D2: float, # D2 Atom-Kreisfrequenz (rad/s)
    Gamma_D2: float    # D2 Linienbreite (rad/s)
) -> float:
    """
    Berechnet die Potentialtiefe U0 (in Joule) für einen einzelnen Strahl
    nach dem "Extended Model"  
    """
 
    
    # Term für die D1-Linie
    term_D1 = (Gamma_D1 / omega_0_D1**3) * \
                (1 / (omega_0_D1 - omega_L) + 1 / (omega_0_D1 + omega_L))
                
    # Term für die D2-Linie
    term_D2 = (2 * Gamma_D2 / omega_0_D2**3) * \
                (1 / (omega_0_D2 - omega_L) + 1 / (omega_0_D2 + omega_L))

    # Präfaktor  
    prefactor = -(pi * c_light**2) / 2
    
    # Intensität im Fokus (I_0 = 2P / (pi * w0^2))
 
    intensity_factor = (2 * P) / (pi * w0**2)

    U0 = prefactor * (term_D1 + term_D2) * intensity_factor
    
    return U0


def calculate_crossed_trap_frequencies(
    U0_1: float,         # Potentialtiefe Strahl 1 (in Joule)
    w0_1: float,         # Strahl-Taille (Waist) Strahl 1 (in Metern)
    zR_1: float,         # Rayleigh-Länge Strahl 1 (in Metern)
    U0_2: float,         # Potentialtiefe Strahl 2 (in Joule)
    w0_2: float,         # Strahl-Taille (Waist) Strahl 2 (in Metern)
    zR_2: float,         # Rayleigh-Länge Strahl 2 (in Metern)
    m: float             # Atommasse (in kg)
) -> dict:
    """
    Berechnet   Fallen-Frequenzen  
    für eine gekreuzte Dipolfalle  

 
    """
    
    abs_U0_1 = abs(U0_1)
    abs_U0_2 = abs(U0_2)
    
    # --- 1. Frequenzen für Strahl 1 (propagiert entlang x) ---
    
    # Radiale Frequenz  
    omega_perp_1_sq = (4 * abs_U0_1) / (m * w0_1**2)
    
    # Axiale Frequenz  
    omega_axial_1_sq = (2 * abs_U0_1) / (m * zR_1**2)

    # --- 2. Frequenzen für Strahl 2 (propagiert entlang y) ---

    # Radiale Frequenz  
    omega_perp_2_sq = (4 * abs_U0_2) / (m * w0_2**2)
    
    # Axiale Frequenz  
    omega_axial_2_sq = (2 * abs_U0_2) / (m * zR_2**2)
    
    
    # Frequenz in x-Richtung  
    omega_x_sq = omega_axial_1_sq + omega_perp_2_sq
    omega_x = math.sqrt(omega_x_sq)
    
    # Frequenz in y-Richtung  
    omega_y_sq = omega_perp_1_sq + omega_axial_2_sq
    omega_y = math.sqrt(omega_y_sq)
    
    # Frequenz in z-Richtung  
    omega_z_sq = omega_perp_1_sq + omega_perp_2_sq
    omega_z = math.sqrt(omega_z_sq)
    
    # --- 4. Mittlere Frequenz ---  
    omega_mean = (omega_x * omega_y * omega_z)**(1/3.0)
    
    # Rückgabe  
    return {
        "omega_x_rad_s": omega_x,
        "omega_y_rad_s": omega_y,
        "omega_z_rad_s": omega_z,
        "omega_mean_rad_s": omega_mean,
        "freq_x_hz": omega_x / (2 * math.pi),
        "freq_y_hz": omega_y / (2 * math.pi),
        "freq_z_hz": omega_z / (2 * math.pi),
        "freq_mean_hz": omega_mean / (2 * math.pi)
    }
