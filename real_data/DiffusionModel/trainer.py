# train.py
import math, time, copy, dataclasses
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR, LambdaLR
)
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def seed_everything(seed: int = 42):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      # keeps CUDA deterministic

# weight initialisation (good defaults for Conv/Linear)
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
# ------------------------------------------------------------------
#  1.  Helpers
# ------------------------------------------------------------------
class EMA:
    """
    Exponential Moving Average of model parameters.
    Call `update()` after every optimiser.step().
    Use `apply_shadow()`/`restore()` to evaluate with the EMA weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.original_params, self.shadow_params = [], []

        # store pointers to the params so they stay in sync
        for p in model.parameters():
            if p.requires_grad:
                self.original_params.append(p)
                self.shadow_params.append(p.detach().clone())

    @torch.no_grad()
    def update(self):
        for p, s in zip(self.original_params, self.shadow_params):
            s.mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    # --- evaluation helpers -------------------------------------------------
    def apply_shadow(self):
        self.backup = [p.detach().clone() for p in self.original_params]
        for p, s in zip(self.original_params, self.shadow_params):
            p.copy_(s)

    def restore(self):
        for p, b in zip(self.original_params, self.backup):
            p.copy_(b)
        del self.backup


# ------------------------------------------------------------------
#  2.  Config dataclass – tweak from the CLI or a YAML/JSON file
# ------------------------------------------------------------------
@dataclasses.dataclass
class TrainCfg:
    # optimiser
    optim_name: str = "AdamW"       # any member of torch.optim
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)     # used by Adam-family
    momentum: float = 0.9           # used by SGD/RMSprop
    # scheduler
    scheduler: Optional[str] = "cosine"   # cosine|cosine_restart|step|exp|plateau|1cycle|none
    warmup_steps: int = 0
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    # misc
    num_epochs: int = 400
    max_grad_norm: Optional[float] = 1.0
    amp: bool = True                # mixed precision
    patience: int = 20              # early-stopping


# ------------------------------------------------------------------
#  3.  Factory helpers
# ------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: TrainCfg) -> optim.Optimizer:
    kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optim_name.lower() in {"adam", "adamw", "adamax", "nadam", "radam"}:
        kwargs.update(betas=cfg.betas)
    if cfg.optim_name.lower() in {"sgd", "rmsprop"}:
        kwargs.update(momentum=cfg.momentum)

    optim_cls = getattr(optim, cfg.optim_name)
    return optim_cls(model.parameters(), **kwargs)


def build_scheduler(optimizer: optim.Optimizer, cfg: TrainCfg,
                    total_steps: int) -> Optional[optim.lr_scheduler._LRScheduler]:
    if cfg.scheduler is None or cfg.scheduler.lower() == "none":
        return None

    name = cfg.scheduler.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    if name == "cosine_restart":
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    if name == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    if name == "exp":
        return ExponentialLR(optimizer, gamma=0.97)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    if name == "1cycle":
        return OneCycleLR(optimizer, max_lr=cfg.lr,
                          total_steps=total_steps, pct_start=0.3, anneal_strategy="cos")
    raise ValueError(f"Unknown scheduler {cfg.scheduler}")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


# ------------------------------------------------------------------
#  4.  Main training loop
# ------------------------------------------------------------------
def fit(model: nn.Module, dataloader, val_loader,
        device: torch.device, cfg: TrainCfg, save_name: str = "best.pt"):
    seed_everything()          # <── NEW
    model.apply(init_weights)  # <── NEW   (remove if your model already inits)
    model.to(device)
    optimizer = build_optimizer(model, cfg)

    total_steps = cfg.num_epochs * len(dataloader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.amp)

    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    best_loss, epochs_no_improve = math.inf, 0
    start = time.time()
    step_per_batch = isinstance(scheduler, (OneCycleLR, LambdaLR))
    for epoch in range(cfg.num_epochs):
        # ---- training ------------------------------------------------------
        model.train()
        running_loss = 0.0
        prog = tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}")
        for step, (batch_x, batch_y) in enumerate(prog, 1):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):

                loss, _ = model(batch_x, batch_y)

            scaler.scale(loss).backward()

            if cfg.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            if scheduler and not step_per_batch:
              scheduler.step()

            if ema:
                ema.update()

            running_loss += loss.item() * batch_x.size(0)
            prog.set_postfix(loss=loss.item(), lr=get_lr(optimizer))

        epoch_loss = running_loss / len(dataloader.dataset)

        # ---- validation ----------------------------------------------------
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(f"[{epoch+1:3d}/{cfg.num_epochs}]  "
              f"train={epoch_loss:.4f}  val={val_loss:.4f}  "
              f"lr={get_lr(optimizer):.2e}  t={elapsed/60:.1f}m")

        if scheduler and cfg.scheduler == "plateau":
            scheduler.step(val_loss)

        # ---- early-stopping ------------------------------------------------
        if val_loss + 1e-6 < best_loss:
            best_loss, epochs_no_improve = val_loss, 0
            torch.save(model.state_dict(), save_name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print("Early stopping")
                break


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, ema: Optional[EMA] = None):
    if ema:
        ema.apply_shadow()
    model.eval()
    loss_sum = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        loss, _ = model(batch_x, batch_y)
        loss_sum += loss.item() * batch_x.size(0)
    if ema:
        ema.restore()
    return loss_sum / len(loader.dataset)


