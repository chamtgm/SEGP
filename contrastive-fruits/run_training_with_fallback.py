import argparse
import subprocess
import shlex
import sys
import os
from datetime import datetime

OOM_KEYWORDS = [
    'out of memory',
    'CUDA out of memory',
    'RuntimeError: CUDA',
]


def run_cmd(cmd, log_path):
    print('Running:', ' '.join(cmd))
    with open(log_path, 'wb') as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            try:
                sys.stdout.buffer.write(line)
            except Exception:
                try:
                    sys.stdout.write(line.decode('utf-8', errors='replace'))
                except Exception:
                    pass
        proc.wait()
        return proc.returncode


def check_oom(log_path):
    txt = ''
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read().lower()
    except Exception:
        return False
    return any(k.lower() in txt for k in OOM_KEYWORDS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fruit-root', required=True)
    p.add_argument('--style-root', required=True)
    p.add_argument('--hvae-ckpt', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[128, 64])
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=0.5)
    args = p.parse_args()

    script = os.path.join(os.path.dirname(__file__), 'train.py')
    py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.venv', 'Scripts', 'python.exe'))
    if not os.path.isfile(py):
        py = sys.executable

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for bs in args.batch_sizes:
        log_dir = os.path.join(os.path.dirname(__file__), 'training_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'train_bs{bs}_{timestamp}.log')

        cmd = [py, script,
               '--fruit-root', args.fruit_root,
               '--style-root', args.style_root,
               '--style-method', 'hvae',
               '--hvae-ckpt', args.hvae_ckpt,
               '--epochs', str(args.epochs),
               '--batch-size', str(bs),
               '--alpha', str(args.alpha),
               '--temperature', str(args.temperature),
               '--device', args.device]

        print('\n=== Attempting training with batch size', bs, '===')
        rc = run_cmd(cmd, log_path)
        if rc == 0:
            print('\nTraining finished successfully with batch size', bs)
            print('Log saved to', log_path)
            return 0
        else:
            print(f'Process exited with code {rc}. Checking log for OOM...')
            if check_oom(log_path):
                print('Detected OOM in log. Will try next smaller batch size if available.')
                continue
            else:
                print('Failure was not OOM. See log:', log_path)
                return rc

    print('All batch-size attempts failed (likely OOM at smallest size).')
    return 1


if __name__ == '__main__':
    sys.exit(main())
