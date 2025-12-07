#!/usr/bin/env python3

from torch.utils.tensorboard import SummaryWriter


class ModifiedTensorBoard:
    """Custom TensorBoard wrapper for PyTorch"""
    
    def __init__(self, **kwargs):
        self.log_dir = kwargs.get('log_dir', 'logs')
        self.step = 1
        self.writer = SummaryWriter(self.log_dir)

    def update_stats(self, **stats):
        """Write stats to TensorBoard"""
        for key, value in stats.items():
            self.writer.add_scalar(f"metrics/{key}", value, self.step)
        self.writer.flush()