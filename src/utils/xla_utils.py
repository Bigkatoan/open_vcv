"""
xla_utils.py — TPU / torch_xla helpers tối ưu cho Kaggle TPU v5e-8.

TPU v5e-8 specs:
    - 8 cores, mỗi core 16GB HBM2e → tổng 128GB
    - Native bfloat16 (nhanh hơn float16, stable hơn)
    - SPMD parallelism (torch_xla 2.x) — phân tán tự động qua 8 cores
    - Không cần GradScaler — bfloat16 không overflow như float16

Cách dùng:
    from src.utils.xla_utils import setup_tpu, get_device, bf16_context, clip_and_step

    setup_tpu()                    # gọi 1 lần ở đầu script
    device = get_device()          # xla:0 trên TPU, cuda/cpu fallback
    model  = model.to(device)
    ...
    with bf16_context():           # bfloat16 autocast (TPU) hoặc float16 (GPU)
        out = model(x)
    clip_and_step(optimizer, model)
    mark_step()

Notes:
    - torch_xla chỉ available trên Kaggle TPU / Google Cloud TPU VM.
    - channels_last KHÔNG hỗ trợ trên TPU — tự động skip.
    - SPMD: torch_xla 2.x tự shard tensors qua 8 cores → batch effective × 8.
    - BF16: dùng XLA_USE_BF16=1 env var + torch.bfloat16 autocast.
"""

from __future__ import annotations
import os
import contextlib
import torch

# ── Detect XLA ────────────────────────────────────────────────────────────────
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

    # SPMD (torch_xla 2.x, v5e-optimized)
    try:
        import torch_xla.experimental.xla_sharding as xs
        import torch_xla.runtime as xr
        _SPMD_AVAILABLE = True
    except ImportError:
        _SPMD_AVAILABLE = False

    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False
    _SPMD_AVAILABLE = False


def xla_available() -> bool:
    return _XLA_AVAILABLE


def spmd_available() -> bool:
    return _SPMD_AVAILABLE


# ── TPU Setup ─────────────────────────────────────────────────────────────────

def setup_tpu(use_bf16: bool = True, use_spmd: bool = True):
    """
    Khởi tạo TPU v5e-8 với các tối ưu tốt nhất.
    Gọi 1 lần ở đầu script trước khi tạo model / dataloader.

    use_bf16 : dùng bfloat16 — native trên v5e, nhanh hơn float32 ~2×
    use_spmd : bật SPMD parallelism — tự shard batch qua 8 cores
    """
    if not _XLA_AVAILABLE:
        return  # GPU/CPU: no-op

    if use_bf16:
        # XLA_USE_BF16=1: tự động cast float32 ops sang bfloat16 trên TPU
        os.environ['XLA_USE_BF16'] = '1'
        # Tắt downcast cảnh báo
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    if use_spmd and _SPMD_AVAILABLE:
        # SPMD mode: 1 process điều khiển 8 cores
        # Tốt hơn multiprocessing cho v5e (ít overhead hơn)
        xr.use_spmd()
        xla_print(f"[TPU] SPMD mode enabled — {xr.global_device_count()} devices")
    else:
        xla_print(f"[TPU] Single-process mode")

    xla_print(f"[TPU] v5e-8 ready | BF16={use_bf16} | SPMD={use_spmd and _SPMD_AVAILABLE}")


def get_device() -> torch.device:
    """
    Trả về device theo thứ tự ưu tiên: TPU → CUDA → CPU.
    Trên Kaggle TPU: trả về xm.xla_device() (xla:0).
    """
    if _XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def device_str() -> str:
    if _XLA_AVAILABLE:
        return 'xla'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def is_tpu() -> bool:
    return _XLA_AVAILABLE


# ── bfloat16 context ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def bf16_context():
    """
    Context manager cho mixed precision:
    - TPU v5e: torch.bfloat16 autocast (native, không overflow)
    - GPU    : torch.float16 autocast (AMP thông thường)
    - CPU    : no-op
    """
    if _XLA_AVAILABLE:
        with torch.autocast('xla', dtype=torch.bfloat16):
            yield
    elif torch.cuda.is_available():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            yield
    else:
        yield


def get_autocast_dtype() -> torch.dtype | None:
    """Trả về dtype dùng cho autocast: bfloat16 (TPU), float16 (GPU), None (CPU)."""
    if _XLA_AVAILABLE:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


# ── Training helpers ───────────────────────────────────────────────────────────

def mark_step():
    """
    BẮT BUỘC gọi sau mỗi optimizer step khi dùng TPU.
    XLA lazy execution: mark_step() flush tất cả ops pending vào hardware.
    Trên GPU/CPU: no-op.
    """
    if _XLA_AVAILABLE:
        xm.mark_step()


def clip_and_step(optimizer, model, scaler=None, max_norm: float = 0.5):
    """
    Gradient clip + optimizer step + mark_step (TPU) / scaler.update (GPU).

    TPU v5e: XLA tự handle bfloat16 scaling — không cần GradScaler.
    GPU    : dùng GradScaler nếu có (AMP float16).
    """
    if _XLA_AVAILABLE:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        xm.optimizer_step(optimizer)
        xm.mark_step()
    elif scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()


def wrap_dataloader(loader, device):
    """
    Wrap DataLoader với MpDeviceLoader trên TPU để pipeline data → XLA device.
    Trên GPU/CPU: trả về loader nguyên.
    """
    if _XLA_AVAILABLE:
        return pl.MpDeviceLoader(loader, device)
    return loader


def shard_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    SPMD: shard model parameters qua 8 TPU cores để data parallelism.
    Tương đương DistributedDataParallel nhưng không cần spawn processes.
    Trên GPU/CPU: no-op.
    """
    if _SPMD_AVAILABLE:
        # Đơn giản nhất: replicate toàn bộ model qua tất cả devices
        # (batch tự động split theo số devices)
        num_devices = xr.global_device_count()
        mesh = xs.Mesh(list(range(num_devices)), (num_devices,), ('batch',))
        for param in model.parameters():
            xs.mark_sharding(param, mesh, (None,))   # replicate params
        xla_print(f"[TPU] Model sharded across {num_devices} cores (data parallel)")
    return model


# ── Distributed helpers ────────────────────────────────────────────────────────

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all TPU cores (mean). GPU: no-op."""
    if _XLA_AVAILABLE:
        return xm.all_reduce(xm.REDUCE_SUM, tensor) / xm.xrt_world_size()
    return tensor


def get_world_size() -> int:
    if _XLA_AVAILABLE:
        try:
            return xr.global_device_count() if _SPMD_AVAILABLE else xm.xrt_world_size()
        except Exception:
            return 1
    return 1


def get_ordinal() -> int:
    if _XLA_AVAILABLE:
        return xm.get_ordinal()
    return 0


def is_master() -> bool:
    if _XLA_AVAILABLE:
        return xm.is_master_ordinal()
    return True


def xla_print(*args):
    """Print chi tu master core (tranh 8 cores print cung luc)."""
    if _XLA_AVAILABLE:
        if xm.is_master_ordinal():
            print(*args)
    else:
        print(*args)

