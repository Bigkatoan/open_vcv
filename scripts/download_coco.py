#!/usr/bin/env python3
"""
Tải COCO 2017 với tốc độ tối đa

Chiến lược tốc độ:
    1. aria2c  — tốt nhất: multi-connection + multi-thread per file (~10-16 kết nối/file)
    2. Python parallel (concurrent.futures) — fallback nếu không có aria2c

Chạy:
    python scripts/download_coco.py
    python scripts/download_coco.py --split train         # chỉ train (18 GB)
    python scripts/download_coco.py --resume              # resume nếu bị ngắt
    python scripts/download_coco.py --connections 16      # số kết nối aria2c
"""

import os
import sys
import argparse
import zipfile
import shutil
import subprocess
import threading
import urllib.request
import concurrent.futures
from pathlib import Path


URLS = {
    'train':       'http://images.cocodataset.org/zips/train2017.zip',
    'val':         'http://images.cocodataset.org/zips/val2017.zip',
    'test':        'http://images.cocodataset.org/zips/test2017.zip',
    'unlabeled':   'http://images.cocodataset.org/zips/unlabeled2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}
SIZES = {'train': '18 GB', 'val': '1 GB', 'test': '6 GB',
         'unlabeled': '19 GB', 'annotations': '241 MB'}

DATA_DIR = Path(__file__).parent.parent / 'data' / 'coco2017'


# ===========================================================================
# Method 1: aria2c (fastest — multi-connection)
# ===========================================================================

def download_aria2c(url: str, dest: Path, connections: int = 16, resume: bool = False):
    """
    aria2c: tải song song N kết nối vào cùng 1 file → saturate bandwidth.
    Install: sudo apt install aria2
    """
    cmd = [
        'aria2c',
        f'--split={connections}',
        f'--max-connection-per-server={connections}',
        '--min-split-size=10M',
        '--continue=true' if resume else '--continue=false',
        '--file-allocation=falloc',
        '--auto-file-renaming=false',     # QUAN TRỌNG: không đổi tên thành .1.zip
        '--allow-overwrite=true',         # cho phép ghi đè nếu đã tồn tại
        '--console-log-level=error',
        '--summary-interval=5',
        '--human-readable=true',
        f'--dir={dest.parent}',
        f'--out={dest.name}',
        url,
    ]
    print(f"  aria2c ({connections} connections)...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("aria2c thất bại")


# ===========================================================================
# Method 2: Python parallel chunked download (fallback)
# ===========================================================================

def _get_filesize(url: str) -> int:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get('Content-Length', 0))


def _download_chunk(url, start, end, chunk_idx, tmp_dir, progress):
    """Tải 1 chunk [start, end] bytes."""
    tmp_file = tmp_dir / f'chunk_{chunk_idx:04d}'
    if tmp_file.exists() and tmp_file.stat().st_size == (end - start + 1):
        progress[chunk_idx] = end - start + 1
        return  # chunk đã có

    req = urllib.request.Request(
        url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp_file, 'wb') as f:
        while True:
            data = r.read(256 * 1024)
            if not data:
                break
            f.write(data)
            progress[chunk_idx] = progress.get(chunk_idx, 0) + len(data)


def _print_progress(progress, total, stop_event, filename):
    import time
    while not stop_event.is_set():
        downloaded = sum(progress.values())
        pct = min(100, int(downloaded * 100 / total)) if total else 0
        bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
        mb_done  = downloaded / 1e6
        mb_total = total / 1e6
        print(f"\r  [{bar}] {pct:3d}%  {mb_done:.0f}/{mb_total:.0f} MB",
              end='', flush=True)
        time.sleep(1)
    print()


def download_parallel(url: str, dest: Path, n_threads: int = 8, resume: bool = False):
    """Fallback: tải song song N chunks bằng Python threads."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest.parent / f'.tmp_{dest.stem}'
    tmp_dir.mkdir(exist_ok=True)

    total = _get_filesize(url)
    if not total:
        # Server không hỗ trợ Range → tải thường
        print(f"  Tải thẳng (server không hỗ trợ multi-thread)...")
        urllib.request.urlretrieve(url, dest)
        return

    chunk_size = max(10 * 1024 * 1024, total // n_threads)  # min 10 MB / chunk
    chunks = []
    pos = 0
    while pos < total:
        end = min(pos + chunk_size - 1, total - 1)
        chunks.append((pos, end))
        pos = end + 1

    print(f"  {n_threads} threads, {len(chunks)} chunks, {total/1e9:.1f} GB...")

    progress = {}
    stop_event = threading.Event()
    prog_thread = threading.Thread(
        target=_print_progress, args=(progress, total, stop_event, dest.name), daemon=True)
    prog_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = [ex.submit(_download_chunk, url, s, e, i, tmp_dir, progress)
                for i, (s, e) in enumerate(chunks)]
        concurrent.futures.wait(futs)
        # Re-raise nếu có lỗi
        for f in futs:
            f.result()

    stop_event.set()
    prog_thread.join()

    # Ghép các chunks lại
    print(f"  Ghép {len(chunks)} chunks...")
    with open(dest, 'wb') as out:
        for i in range(len(chunks)):
            chunk_file = tmp_dir / f'chunk_{i:04d}'
            with open(chunk_file, 'rb') as cf:
                shutil.copyfileobj(cf, out, length=4 * 1024 * 1024)

    shutil.rmtree(tmp_dir)


# ===========================================================================
# Download dispatcher
# ===========================================================================

def download(url: str, dest: Path, connections: int, resume: bool):
    if dest.exists():
        expected = _get_filesize(url)
        if expected > 0 and dest.stat().st_size == expected:
            print(f"  Đã có: {dest.name} — bỏ qua")
            return

    has_aria2c = shutil.which('aria2c') is not None

    if has_aria2c:
        download_aria2c(url, dest, connections, resume)
    else:
        print(f"  aria2c không có → dùng Python {connections}-thread download")
        print(f"  (Cài aria2c để tốt hơn: sudo apt install aria2)")
        download_parallel(url, dest, n_threads=connections, resume=resume)


# ===========================================================================
# Extract
# ===========================================================================

def extract(zip_path: Path, out_dir: Path):
    print(f"  Giải nén {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        total   = len(members)
        for i, name in enumerate(members, 1):
            z.extract(name, out_dir)
            if i % 20000 == 0 or i == total:
                print(f"\r  {i:,}/{total:,} files", end='', flush=True)
    print()


# ===========================================================================
# Verify
# ===========================================================================

def verify(data_dir: Path):
    print("\n[Verify]")
    for split in ['train2017', 'val2017']:
        d = data_dir / split
        if not d.exists():
            print(f"  WARNING: {d} không tồn tại!")
            continue
        n = len(list(d.glob('*.jpg')))
        print(f"  {split}: {n:,} ảnh")

    ann = data_dir / 'annotations'
    if ann.exists():
        print(f"  annotations: {len(list(ann.glob('*.json')))} files")

    print(f"\n✓ COCO 2017 tại: {data_dir}")
    print("  train2017/ — 118,287 ảnh")
    print("  val2017/   — 5,000 ảnh")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', nargs='+',
                        default=['train', 'val', 'annotations'],
                        choices=list(URLS.keys()))
    parser.add_argument('--out-dir',     type=str, default=str(DATA_DIR))
    parser.add_argument('--connections', type=int, default=16,
                        help='Số kết nối song song (default: 16)')
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--keep-zip',    action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_aria2c = shutil.which('aria2c') is not None
    method = f"aria2c ({args.connections} conn/file)" if has_aria2c \
             else f"Python parallel ({args.connections} threads)"
    print(f"Download method: {method}")
    print(f"Target: {out_dir}")
    print(f"Splits: {args.split}  (~{'  +  '.join(SIZES[s] for s in args.split)})\n")

    if not has_aria2c:
        print("TIP: Cài aria2c để tốc độ cao nhất:")
        print("     sudo apt install aria2\n")

    for split in args.split:
        url  = URLS[split]
        name = f'{split}2017.zip' if split != 'annotations' \
               else 'annotations_trainval2017.zip'
        dest = out_dir / name

        print(f"\n[{split.upper()}] ({SIZES[split]})")
        download(url, dest, args.connections, args.resume)
        extract(dest, out_dir)

        if not args.keep_zip:
            dest.unlink()
            print(f"  Đã xóa {dest.name}")

    verify(out_dir)


if __name__ == '__main__':
    main()
