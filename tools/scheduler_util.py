import torch
from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter
from torch.optim.lr_scheduler import ReduceLROnPlateau, EPOCH_DEPRECATION_WARNING
import warnings


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            patience=0,
            eval_period=1,
            last_epoch: int = -1,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.last_epoch = last_epoch
        self.eval_period = eval_period
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        super().__init__(optimizer, mode='max', factor=0.1, patience=patience, threshold=1e-4,
                         threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if epoch <= self.warmup_iters:
            warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters,
                                                       self.warmup_factor)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor

        else:
            if (epoch-self.warmup_iters-1) % self.eval_period == 0:
                if self.is_better(current, self.best):
                    self.best = current
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                if self.in_cooldown:
                    self.cooldown_counter -= 1
                    self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

                if self.num_bad_epochs > self.patience:
                    self._reduce_lr(epoch)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
