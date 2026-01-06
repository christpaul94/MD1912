import torch
import triton
import triton.language as tl
from pykeops.torch import LazyTensor

# ============================================================================
# 1. PYTORCH (Naive, Loop, & Optimized)
# ============================================================================
#@torch.compile(mode="reduce-overhead")
def loop_DxN(q, r0, c, chunk=None):
    """
    Langsame Python-Schleife über N. 
    Dient als absolute Baseline (sehr langsam bei großem N).
    """
    D, N = q.shape
    r0_2 = r0**2
    factor = c / r0_2
    forces = torch.zeros_like(q)
    
    # Wir berechnen hier keine Potential-Rückgabe für den Benchmark-Vergleich,
    # um die Signatur (q, r0, c, chunk) einheitlich zu halten.
    for i in range(N):
        q_i = q[:, i:i+1]
        diff = q_i - q
        r_sq = (diff**2).sum(dim=0)
        exp_term = torch.exp(-r_sq / (2 * r0_2))
        forces[:, i] = (diff * (exp_term * factor).unsqueeze(0)).sum(dim=1)
        
    return forces

#@torch.compile(mode="reduce-overhead")
def naive_DxN(q, r0, c, chunk=None):
    """
    Vollständiges O(N^2) Broadcasting. 
    Warnung: Führt bei großem N sofort zu Out-Of-Memory (OOM).
    """
    D, N = q.shape
    r0_2 = r0**2
    factor = c / r0_2
    
    # (D, N, 1) - (D, 1, N) -> (D, N, N) -> OOM Gefahr!
    diff = q.unsqueeze(2) - q.unsqueeze(1)
    r_sq = (diff**2).sum(dim=0) 
    
    exp_term = torch.exp(-r_sq / (2 * r0_2))
    force_vecs = diff * (exp_term * factor).unsqueeze(0)
    forces = force_vecs.sum(dim=2)
    
    return forces
    
@torch.compile(mode="reduce-overhead")
def pytorch_chunked_DxN(q, sigma, V0, chunk=1024):
    D, N = q.shape
    sigma_sq = sigma**2
    prefactor = V0 / sigma_sq
    inv_width = -1 / (2 * sigma_sq)
    forces = torch.empty_like(q)

    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        q_chunk = q[:, i:end]
        diff = q_chunk[:, :, None] - q[:, None, :] 
        r_sq = (diff**2).sum(dim=0)
        force_mag = torch.exp(inv_width * r_sq) * prefactor
        forces[:, i:end] = (diff * force_mag[None, :, :]).sum(dim=2)
    return forces

# ============================================================================
# 2. TRITON (Simple & Tiled)
# ============================================================================

@triton.jit
def _kernel_simple(ptr_q, ptr_forces, N, sigma_sq, V0, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    prefactor = V0 / sigma_sq; inv_width = -1.0 / (2.0 * sigma_sq)
    
    off_x = pid; off_y = N + pid; off_z = 2 * N + pid
    q_i_x = tl.load(ptr_q + off_x); q_i_y = tl.load(ptr_q + off_y); q_i_z = tl.load(ptr_q + off_z)
    acc_fx = 0.0; acc_fy = 0.0; acc_fz = 0.0
    
    for j_start in range(0, N, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE); mask = (j_start + offs) < N
        q_j_x = tl.load(ptr_q + j_start + offs, mask=mask, other=0.0)
        q_j_y = tl.load(ptr_q + N + j_start + offs, mask=mask, other=0.0)
        q_j_z = tl.load(ptr_q + 2 * N + j_start + offs, mask=mask, other=0.0)
        
        dx = q_i_x - q_j_x; dy = q_i_y - q_j_y; dz = q_i_z - q_j_z
        r_sq = dx*dx + dy*dy + dz*dz
        mag = tl.where(mask, tl.exp(inv_width * r_sq) * prefactor, 0.0)
        acc_fx += tl.sum(dx * mag); acc_fy += tl.sum(dy * mag); acc_fz += tl.sum(dz * mag)

    tl.store(ptr_forces + off_x, acc_fx); tl.store(ptr_forces + off_y, acc_fy); tl.store(ptr_forces + off_z, acc_fz)

@triton.jit
def _kernel_tiled(ptr_q, ptr_forces, N, sigma_sq, V0, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M); mask_m = offs_m < N
    prefactor = V0 / sigma_sq; inv_width = -1.0 / (2.0 * sigma_sq)

    q_i_x = tl.load(ptr_q + offs_m, mask=mask_m, other=0.0)
    q_i_y = tl.load(ptr_q + N + offs_m, mask=mask_m, other=0.0)
    q_i_z = tl.load(ptr_q + 2 * N + offs_m, mask=mask_m, other=0.0)

    acc_fx = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_fy = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_fz = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N); mask_n = offs_n < N
        q_j_x = tl.load(ptr_q + offs_n, mask=mask_n, other=0.0)
        q_j_y = tl.load(ptr_q + N + offs_n, mask=mask_n, other=0.0)
        q_j_z = tl.load(ptr_q + 2 * N + offs_n, mask=mask_n, other=0.0)

        dx = q_i_x[:, None] - q_j_x[None, :]
        dy = q_i_y[:, None] - q_j_y[None, :]
        dz = q_i_z[:, None] - q_j_z[None, :]
        
        r_sq = dx*dx + dy*dy + dz*dz
        f_mag = tl.where(mask_n[None, :], tl.exp(inv_width * r_sq) * prefactor, 0.0)

        acc_fx += tl.sum(dx * f_mag, axis=1)
        acc_fy += tl.sum(dy * f_mag, axis=1)
        acc_fz += tl.sum(dz * f_mag, axis=1)

    tl.store(ptr_forces + offs_m, acc_fx, mask=mask_m)
    tl.store(ptr_forces + N + offs_m, acc_fy, mask=mask_m)
    tl.store(ptr_forces + 2 * N + offs_m, acc_fz, mask=mask_m)

def triton_simple_DxN(q, sigma, V0, chunk=None):
    if not q.is_contiguous(): q = q.contiguous()
    D, N = q.shape
    forces = torch.empty_like(q)
    _kernel_simple[(N,)](q, forces, N, sigma**2, V0, BLOCK_SIZE=1024, num_warps=4)
    return forces

def triton_tiled_DxN(q, sigma, V0, chunk=None):
    if not q.is_contiguous(): q = q.contiguous()
    D, N = q.shape
    forces = torch.empty_like(q)
    BLOCK_M = 128
    _kernel_tiled[(triton.cdiv(N, BLOCK_M),)](q, forces, N, sigma**2, V0, BLOCK_M=BLOCK_M, BLOCK_N=128, num_warps=4, num_stages=3)
    return forces

# ============================================================================
# 3. KEOPS
# ============================================================================

def keops_DxN(q, sigma, V0, chunk=None):
    q_NxD = q.T.contiguous()
    x_i = LazyTensor(q_NxD[:, None, :])
    x_j = LazyTensor(q_NxD[None, :, :])
    r_sq = (x_i - x_j).sqnorm2()
    pot = V0 * (-r_sq / (2 * sigma**2)).exp()
    forces_NxD = (pot * (x_i - x_j) / sigma**2).sum(1)
    return forces_NxD.T.contiguous()
