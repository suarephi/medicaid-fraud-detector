"""Robust chunked downloader with stall detection and retry."""
import os
import sys
import time
import urllib.request

URL = "https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet"
DEST = "data/medicaid-provider-spending.parquet"
DEST_TMP = DEST + ".tmp"
CHUNK = 256 * 1024  # 256KB chunks - smaller for faster stall detection
STALL_TIMEOUT = 30  # seconds without progress = stall
MAX_RETRIES = 5


def download_attempt():
    """Single download attempt using urllib (no requests dependency issues)."""
    req = urllib.request.Request(URL)
    req.add_header("User-Agent", "Mozilla/5.0")

    resp = urllib.request.urlopen(req, timeout=60)
    total = int(resp.headers.get("Content-Length", 0))
    print(f"Content-Length: {total:,} bytes ({total / 1024 / 1024:.0f} MB)")

    downloaded = 0
    start = time.time()
    last_progress_time = start
    last_report = start

    with open(DEST_TMP, "wb") as f:
        while True:
            try:
                chunk = resp.read(CHUNK)
            except Exception as e:
                raise IOError(f"Read error at {downloaded:,} bytes: {e}")

            if not chunk:
                break

            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            last_progress_time = now

            if now - last_report >= 15:
                elapsed = now - start
                speed = downloaded / elapsed / 1024 / 1024
                pct = downloaded / total * 100 if total else 0
                remaining = (total - downloaded) / (downloaded / elapsed) if downloaded else 0
                print(f"  {downloaded / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MB "
                      f"({pct:.1f}%) {speed:.1f} MB/s ETA {remaining:.0f}s")
                last_report = now
                sys.stdout.flush()

    return downloaded, total


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data", exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*50}")
        print(f"Download attempt {attempt}/{MAX_RETRIES}")
        print(f"{'='*50}")

        if os.path.exists(DEST_TMP):
            os.remove(DEST_TMP)

        try:
            downloaded, total = download_attempt()

            fsize = os.path.getsize(DEST_TMP)
            print(f"\nDownloaded: {fsize:,} bytes")

            if total and fsize != total:
                print(f"SIZE MISMATCH: expected {total:,}, got {fsize:,}")
                raise IOError("Incomplete download")

            # Verify parquet
            with open(DEST_TMP, "rb") as f:
                header = f.read(4)
                f.seek(-4, 2)
                footer = f.read(4)

            if header != b"PAR1" or footer != b"PAR1":
                raise IOError(f"Bad parquet: header={header} footer={footer}")

            print("Parquet magic bytes OK")
            os.rename(DEST_TMP, DEST)
            print(f"SUCCESS: {DEST}")
            return 0

        except Exception as e:
            print(f"\nFailed: {e}")
            if os.path.exists(DEST_TMP):
                sz = os.path.getsize(DEST_TMP)
                print(f"Partial: {sz:,} bytes ({sz / 1024 / 1024:.0f} MB)")
                os.remove(DEST_TMP)

            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    print("All retries exhausted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
