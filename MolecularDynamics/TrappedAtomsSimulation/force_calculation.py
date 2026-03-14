from pykeops.torch import LazyTensor
import torch
import triton
import triton.language as tl
import numpy as np
import math

def calculate_interaction_strength(
    r0_factor: float,
    a_s_bohr: float = 98.98
) -> tuple[float, float]:
    """
    Berechnet r0_phys und C_phys

    Args:
        r0_factor: faktor
        a_s_bohr: s-Wellen-Streulänge in Bohr-Radien.
                   
    Returns:
        Tupel (r0_phys, C_phys)  
    """
    
    HBAR =   6.62607015e-34 / (2 * math.pi) # J·s
    A0 = 5.29177e-11   # m (Bohr-Radius)
    MASS_RB87 = 86.909 * 1.66054e-27 # kg 
    
    # Streulänge berechnen
    a_phys = a_s_bohr * A0
    
    # Effektive Reichweite des Potenzials berechnen
    r0_phys = a_phys * r0_factor
    
    # Formel für C_phys  
    numerator_phys = 2 * HBAR**2 * a_phys
    denominator_phys = MASS_RB87 * math.sqrt(2 * math.pi) * (r0_phys**3)
    C_phys = numerator_phys / denominator_phys
    
    return r0_phys, C_phys

import torch

#@torch.compile(mode="max-autotune")
def pair_torch_simple_fp(positions: torch.Tensor, r0: float, c: float):
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    r_ij_sq = (diff ** 2).sum(dim=-1)#.clamp_min((10**-6 * r0)**2)
    exp_term = torch.exp(-r_ij_sq / (2 * r0**2))
    forces_magnitude = exp_term * (c / r0**2)
    forces = (forces_magnitude)[..., None] * diff
    total_forces = forces.sum(dim=1)
    total_potential = 0.5 * ((exp_term * c).sum() - N * c)
    return total_forces, total_potential

# ========================================================================
@torch.compile(mode="max-autotune")
def compiled_chunk_step(x_i_chunk, x_j_all, r0_2, c):
    diff = x_i_chunk[:, None, :] - x_j_all[None, :, :]
    r_ij_sq = (diff ** 2).sum(dim=-1)
    exp_term = torch.exp(-r_ij_sq / (2 * r0_2))
    forces_magnitude = exp_term * (c / r0_2)
    forces_matrix = forces_magnitude[..., None] * diff
    forces_chunk = forces_matrix.sum(dim=1)
    potential_sum_i_chunk = (exp_term * c).sum(dim=1)
    return forces_chunk, potential_sum_i_chunk


def pair_torch_chunk_fp(
    positions: torch.Tensor,
    r0: float,
    c: float,
    chunk_size: int = 2048 * 2
):
    N = positions.shape[0]
    r0_2 = r0**2
    forces_list = []
    potential_sum_list = []

    for x_i_chunk in positions.split(chunk_size):
        forces_chunk, potential_sum_i_chunk = compiled_chunk_step(
            x_i_chunk, positions, r0_2, c
        )
        forces_list.append(forces_chunk.clone())
        potential_sum_list.append(potential_sum_i_chunk.clone())

    forces = torch.cat(forces_list, dim=0)
    potential_sum_i = torch.cat(potential_sum_list, dim=0)
    total_sum = potential_sum_i.sum()
    total_potential = 0.5 * (total_sum - N * c)
    return forces, total_potential


# ========================================================================

def pair_keops_fp(positions: torch.Tensor, r: float, c: float):
    N = positions.shape[0]
    r0_2 = r**2
    x_i = LazyTensor(positions[:, None, :])
    x_j = LazyTensor(positions[None, :, :])

    diff = x_i - x_j
    r_sq = diff.sqnorm2()

    exp_term = (-r_sq / (2 * r0_2)).exp()
    potential_ij = c * exp_term

    # potential_sum_i enthält die Selbst-Interaktion (V_ii = c * exp(0) = c)
    potential_sum_i = potential_ij.sum_reduction(axis=1)
    total_sum = potential_sum_i.sum()

    # Korrektur für Selbst-Interaktion V_ii = c und Doppeltzählung
    total_potential = 0.5 * (total_sum - N * c)

    forces = (potential_ij * diff / r0_2).sum_reduction(axis=1)
    return forces, total_potential



# ========================================================================
# --- Triton V1   ---
# ========================================================================
@triton.jit
def triton_kernel_v1(
    ptr_positions, ptr_forces, ptr_potential_sum_i,
    N: tl.int32, r0: tl.float32, c: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(axis=0)
    pos_i_x = tl.load(ptr_positions + i * 3 + 0)
    pos_i_y = tl.load(ptr_positions + i * 3 + 1)
    pos_i_z = tl.load(ptr_positions + i * 3 + 2)
    force_acc_x, force_acc_y, force_acc_z = 0.0, 0.0, 0.0
    potential_acc = 0.0
    r0_2 = r0 * r0
    for j_tile in range(0, N, BLOCK_SIZE):
        offs_j = j_tile + tl.arange(0, BLOCK_SIZE)
        mask_j = offs_j < N
        # Pointer-Berechnung in JEDER Iteration
        ptr_j = ptr_positions + offs_j * 3
        pos_j_x = tl.load(ptr_j + 0, mask=mask_j)
        pos_j_y = tl.load(ptr_j + 1, mask=mask_j)
        pos_j_z = tl.load(ptr_j + 2, mask=mask_j)
        diff_x = pos_i_x - pos_j_x
        diff_y = pos_i_y - pos_j_y
        diff_z = pos_i_z - pos_j_z
        r_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
        r_sq = tl.where(i == offs_j, float('inf'), r_sq)
        exp_term = tl.exp(-r_sq / (2.0 * r0_2))
        exp_term = tl.where(mask_j, exp_term, 0.0)
        potential_ij = c * exp_term
        potential_acc += tl.sum(potential_ij)
        force_scalar = potential_ij / r0_2
        force_acc_x += tl.sum(force_scalar * diff_x)
        force_acc_y += tl.sum(force_scalar * diff_y)
        force_acc_z += tl.sum(force_scalar * diff_z)
    tl.store(ptr_forces + i * 3 + 0, force_acc_x)
    tl.store(ptr_forces + i * 3 + 1, force_acc_y)
    tl.store(ptr_forces + i * 3 + 2, force_acc_z)
    tl.store(ptr_potential_sum_i + i, potential_acc)

def pair_triton_fp_v1(positions: torch.Tensor, r: float, c: float):
    N = positions.shape[0]
    forces = torch.empty_like(positions)
    potential_sum_i = torch.empty(N, device=positions.device, dtype=positions.dtype)
    grid = (N,)
    triton_kernel_v1[grid](
        positions, forces, potential_sum_i, N, r, c, BLOCK_SIZE=2048
    )
    total_sum = potential_sum_i.sum()
    total_potential = total_sum * 0.5
    return forces, total_potential

# ========================================================================
# --- Triton V2   ---
# ========================================================================
@triton.jit
def triton_kernel_v2(
    ptr_positions, ptr_forces, ptr_potential_sum_i,
    N: tl.int32, r0: tl.float32, c: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(axis=0)
    pos_i_x = tl.load(ptr_positions + i * 3 + 0)
    pos_i_y = tl.load(ptr_positions + i * 3 + 1)
    pos_i_z = tl.load(ptr_positions + i * 3 + 2)
    force_acc_x, force_acc_y, force_acc_z = 0.0, 0.0, 0.0
    potential_acc = 0.0
    r0_2 = r0 * r0

    # Pointer-Initialisierung  
    offs_j_initial = tl.arange(0, BLOCK_SIZE)
    ptr_j_x = ptr_positions + offs_j_initial * 3 + 0
    ptr_j_y = ptr_positions + offs_j_initial * 3 + 1
    ptr_j_z = ptr_positions + offs_j_initial * 3 + 2
    
    for j_tile in range(0, N, BLOCK_SIZE):
        offs_j = j_tile + offs_j_initial
        mask_j = offs_j < N
        
        # Laden von den Pointern
        pos_j_x = tl.load(ptr_j_x, mask=mask_j)
        pos_j_y = tl.load(ptr_j_y, mask=mask_j)
        pos_j_z = tl.load(ptr_j_z, mask=mask_j)
        
        diff_x = pos_i_x - pos_j_x
        diff_y = pos_i_y - pos_j_y
        diff_z = pos_i_z - pos_j_z
        r_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
        r_sq = tl.where(i == offs_j, float('inf'), r_sq)
        exp_term = tl.exp(-r_sq / (2.0 * r0_2))
        exp_term = tl.where(mask_j, exp_term, 0.0)
        potential_ij = c * exp_term
        potential_acc += tl.sum(potential_ij)
        force_scalar = potential_ij / r0_2
        force_acc_x += tl.sum(force_scalar * diff_x)
        force_acc_y += tl.sum(force_scalar * diff_y)
        force_acc_z += tl.sum(force_scalar * diff_z)
        
        # Pointer-Inkrementierung am Ende der Schleife
        ptr_j_x += BLOCK_SIZE * 3
        ptr_j_y += BLOCK_SIZE * 3
        ptr_j_z += BLOCK_SIZE * 3

    tl.store(ptr_forces + i * 3 + 0, force_acc_x)
    tl.store(ptr_forces + i * 3 + 1, force_acc_y)
    tl.store(ptr_forces + i * 3 + 2, force_acc_z)
    tl.store(ptr_potential_sum_i + i, potential_acc)

def pair_triton_fp_v2(positions: torch.Tensor, r: float, c: float):
    N = positions.shape[0]
    forces = torch.empty_like(positions)
    potential_sum_i = torch.empty(N, device=positions.device, dtype=positions.dtype)
    grid = (N,)
    triton_kernel_v2[grid](
        positions, forces, potential_sum_i, N, r, c, BLOCK_SIZE=2048
    )
    total_sum = potential_sum_i.sum()
    total_potential = total_sum * 0.5
    return forces, total_potential


import torch
from pykeops.torch import LazyTensor

def pair_force_keops_NxD(q: torch.Tensor, sigma: float, V0: float, chunk=None) -> torch.Tensor:
    """
    Berechnet die paarweisen Kräfte basierend auf einem Gauß-Potential.
    Erwartet q im Format (N, D).
    """
    # Form: (N, 1, D) und (1, N, D)
    x_i = LazyTensor(q[:, None, :])
    x_j = LazyTensor(q[None, :, :])
    
    # Quadrierter Abstand
    r_sq = (x_i - x_j).sqnorm2()
    
    # Kraftberechnung: -nabla V
    pot_term = V0 * (-r_sq / (2 * sigma**2)).exp()
    forces_NxD = (pot_term * (x_i - x_j) / sigma**2).sum(dim=1)
    
    return forces_NxD


def pair_potential_keops_NxD(q: torch.Tensor, sigma: float, V0: float, chunk=None) -> torch.Tensor:
    """
    Berechnet die gesamte potentielle Energie des Systems.
    Erwartet q im Format (N, D).
    """
    N = q.size(0)
    
    x_i = LazyTensor(q[:, None, :])
    x_j = LazyTensor(q[None, :, :])
    
    r_sq = (x_i - x_j).sqnorm2()
    
    # Gauß-Potential Matrix
    pot_matrix = V0 * (-r_sq / (2 * sigma**2)).exp()
    
    # Summiere erst über alle Nachbarn (dim=1), dann über alle Teilchen
    total_pot_with_self = pot_matrix.sum(dim=1).sum()
    
    # Korrektur: Selbstwechselwirkung abziehen (N * V0) und Doppelzählung halbieren
    total_pot = 0.5 * (total_pot_with_self - (N * V0))
    
    return total_pot
