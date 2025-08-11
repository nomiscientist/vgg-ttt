# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import time

import torch


class Timer:
    """Context mananager that measures the elapsed time of a code block with CUDA synchronization."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self

        # Synchronize CUDA if available to ensure accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self  # Return self to capture elapsed time later

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        # Synchronize CUDA again to ensure all GPU operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

    def get(self):
        """Returns the elapsed time in seconds."""
        if not self.enabled:
            return 0.0

        return self.elapsed_time


def bytes_to_gb(bytes: int):
    return bytes / 1024**3


class PeakMemory:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self

        torch.cuda.reset_peak_memory_stats()
        self.start_memory = torch.cuda.max_memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = torch.cuda.max_memory_allocated()

    def get(self):
        if not self.enabled:
            return {}

        return {
            "addtional_memory_gb": bytes_to_gb(self.end_memory - self.start_memory),
            "total_peak_memory_gb": bytes_to_gb(self.end_memory),
        }
