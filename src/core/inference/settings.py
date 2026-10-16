"""Bound CPU parallelism consistently across the Python runtimes."""
import os

CPU_THREADS = int(os.environ.get('M38_CPU_THREADS', '4'))
if not 1 <= CPU_THREADS <= 64:
    raise ValueError('M38_CPU_THREADS must be in 1..64')
