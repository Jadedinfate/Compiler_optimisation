#!/usr/bin/env python3
"""
download_devign.py
Downloads the authentic Devign C/C++ Vulnerability Dataset (with full function code)
from Hugging Face into data/devign/.
"""

import os
import sys
import urllib.request
import json
from pathlib import Path

DATA_DIR = Path("data/devign")
FILES = {
    "train_devign.json": "https://huggingface.co/datasets/AISE-TUDelft/ML4SE23_G6_Devign/resolve/main/train_devign.json",
    "valid_devign.json": "https://huggingface.co/datasets/AISE-TUDelft/ML4SE23_G6_Devign/resolve/main/valid_devign.json",
    "test_devign.json": "https://huggingface.co/datasets/AISE-TUDelft/ML4SE23_G6_Devign/resolve/main/test_devign.json",
}

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[*] File already exists ({dest_path.stat().st_size:,} bytes): {dest_path}")
        return

    print(f"[*] Downloading {url} -> {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r    {downloaded:,} / {total_size:,} bytes ({percent:.1f}%)", end="", flush=True)
        print()
    print(f"[+] Download complete: {dest_path} ({dest_path.stat().st_size:,} bytes)")

def verify_dataset():
    print("\n[*] Verifying downloaded dataset:")
    for fname in FILES.keys():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"[!] Missing {fpath}")
            continue
        
        count = 0
        vuln_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    count += 1
                    if obj.get("target") == 1:
                        vuln_count += 1
                except Exception:
                    pass
        print(f"    - {fname}: {count:,} samples | Vulnerable: {vuln_count:,} ({vuln_count/max(count,1)*100:.1f}%)")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        download_file(url, DATA_DIR / fname)
    verify_dataset()

if __name__ == "__main__":
    main()
