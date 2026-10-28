"""Supervise only this experiment, inside its own memory-limited cgroup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3


def available_ram():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def gpu_used_mib():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True)
    return max(float(v) for v in result.stdout.splitlines())


def main():
    memory_max_gib = int(os.environ.get('VLADBUFF_RAM_MAX_GIB', '10'))
    memory_high_gib = int(os.environ.get('VLADBUFF_RAM_HIGH_GIB', str(memory_max_gib - 1)))
    if not 1 <= memory_high_gib < memory_max_gib <= 10:
        raise ValueError('Experiment RAM budget must be within the validated 10 GiB ceiling')
    cgroup = Path('/sys/fs/cgroup' + Path('/proc/self/cgroup').read_text().strip().split(':')[-1])
    if (cgroup / 'memory.max').read_text().strip() != str(memory_max_gib * GIB):
        raise RuntimeError(f'Missing {memory_max_gib} GiB experiment cgroup hard limit')
    if (cgroup / 'memory.high').read_text().strip() != str(memory_high_gib * GIB):
        raise RuntimeError('Missing experiment memory.high reclaim threshold')
    if (cgroup / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Experiment swap must be disabled')
    if available_ram() < (memory_max_gib + 4) * GIB:
        raise RuntimeError(f'Need at least {memory_max_gib + 4} GiB currently available RAM ({memory_max_gib} GiB budget + 4 GiB reserve)')
    if gpu_used_mib() > 4 * 1024:
        raise RuntimeError('GPU is occupied; refusing to overlap another large GPU task')
    report = Path(os.environ.get('TRAIN_TERMINAL_RUN_DIR', '/tmp')) / 'resource-budget.json'
    state = {'memory_max_gib': memory_max_gib, 'memory_high_gib': memory_high_gib, 'swap_max_bytes': 0,
             'torch_allocator_max_gib': 38, 'device_gpu_stop_gib': 44,
             'system_ram_reserve_gib': 4, 'peak_cgroup_gib': 0,
             'peak_device_gpu_gib': 0, 'minimum_available_ram_gib': available_ram() / GIB,
             'cgroup': str(cgroup), 'status': 'running'}
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    interrupted = False

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = True
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    deadline = None
    try:
        while child.poll() is None:
            ram = available_ram() / GIB
            gpu = gpu_used_mib() / 1024
            peak = int((cgroup / 'memory.peak').read_text()) / GIB
            state.update(peak_cgroup_gib=max(state['peak_cgroup_gib'], peak),
                         peak_device_gpu_gib=max(state['peak_device_gpu_gib'], gpu),
                         minimum_available_ram_gib=min(state['minimum_available_ram_gib'], ram),
                         updated=time.time())
            reason = 'system available RAM below 4 GiB' if ram < 4 else 'device GPU usage above 44 GiB' if gpu > 44 else None
            if reason and not interrupted:
                state['stop_reason'] = reason
                print('Resource guard: ' + reason, flush=True)
                interrupt(signal.SIGINT, None)
            if interrupted and deadline is None:
                deadline = time.monotonic() + 30
            if deadline and time.monotonic() > deadline and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            report.write_text(json.dumps(state, indent=2))
            time.sleep(1)
    finally:
        # A monitoring failure must not leave an unsupervised training process.
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child.returncode or interrupted:
            # A failed supervisor must not orphan its evaluation/loader children.
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        state.update(status='completed' if child.returncode == 0 and not interrupted else 'stopped',
                     exit_code=child.returncode, ended=time.time())
        report.write_text(json.dumps(state, indent=2))
    return child.returncode if child.returncode else 1 if interrupted else 0


if __name__ == '__main__':
    raise SystemExit(main())
