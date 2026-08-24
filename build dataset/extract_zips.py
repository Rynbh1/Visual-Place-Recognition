#!/usr/bin/env python3
"""
Extract all zip files in a directory using multiple parallel workers.
Supports resuming: compares file count in zip vs already extracted files.
  - All files present  → skip
  - Partial extraction → resume with unzip -n (never overwrite existing)
  - Nothing extracted  → fresh extraction with unzip -o
"""

import sys
import os
import subprocess
import argparse
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def count_zip_files(zip_path):
    """Count non-directory entries in a zip (reads central directory only, fast)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return sum(1 for info in zf.infolist() if not info.is_dir())
    except Exception:
        return -1


def count_extracted_files(subdir):
    """Count files already present in the expected output subdirectory."""
    if not subdir.exists():
        return 0
    return sum(1 for f in subdir.rglob("*") if f.is_file())


def make_progress_bar(percentage, width=20):
    percentage = max(0.0, min(1.0, percentage))
    filled = int(round(width * percentage))
    bar = "█" * filled + "░" * (width - filled)
    return bar



def extract_zip(args):
    zip_path, output_dir, show_progress = args
    stem = zip_path.stem
    subdir = Path(output_dir) / stem
    start = time.time()

    try:
        total_in_zip = count_zip_files(zip_path)
        already_extracted = count_extracted_files(subdir)

        if total_in_zip < 0:
            return zip_path.name, "FAIL", 0.0, "could not read zip central directory"

        if already_extracted >= total_in_zip:
            return zip_path.name, "SKIP", 0.0, f"{already_extracted}/{total_in_zip} files already present"

        # Use -n (never overwrite) to resume a partial extraction,
        # or -o (overwrite) for a fresh one.
        flag = "-n" if already_extracted > 0 else "-o"
        status_prefix = "RESUME" if already_extracted > 0 else "EXTRACT"

        if not show_progress:
            # Silent execution for multiple workers to avoid inter-process terminal clutter
            print(f"[{status_prefix}] Starting {zip_path.name} ({already_extracted}/{total_in_zip} files)...", flush=True)
            cmd = ["unzip", flag, "-q", str(zip_path), "-d", str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start
            if result.returncode not in (0, 1):
                return zip_path.name, "FAIL", elapsed, result.stderr.strip()
        else:
            # Interactive progress bar for single-worker mode
            cmd = ["unzip", flag, str(zip_path), "-d", str(output_dir)]
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

            extracted_count = already_extracted
            last_update = time.time()
            last_progress_print = time.time()

            # Read stdout line by line to track progress
            for line in process.stdout:
                if any(x in line for x in ("extracting:", "inflating:", "linking:")):
                    extracted_count += 1
                    current_time = time.time()
                    if current_time - last_update >= 0.1 or extracted_count == total_in_zip:
                        last_update = current_time
                        pct = extracted_count / total_in_zip if total_in_zip > 0 else 0.0
                        elapsed = current_time - start
                        rate = (extracted_count - already_extracted) / elapsed if elapsed > 0 else 0.0
                        rate_str = f"{rate:.1f} f/s" if rate > 0 else "- f/s"
                        rem_files = total_in_zip - extracted_count
                        eta = rem_files / rate if rate > 0 else 0
                        eta_str = f"ETA {format_eta(eta)}" if rate > 0 else "ETA -"

                        bar = make_progress_bar(pct, width=20)
                        print(
                            f"\r[{status_prefix}] {zip_path.name}: [{bar}] {pct*100:5.1f}% "
                            f"({extracted_count}/{total_in_zip} files) | {rate_str} | {eta_str}",
                            end="",
                            flush=True
                        )

            returncode = process.wait()
            stderr_content = process.stderr.read()
            elapsed = time.time() - start

            print()  # Finalize progress bar line

            if returncode not in (0, 1):
                return zip_path.name, "FAIL", elapsed, stderr_content.strip()

        final_count = count_extracted_files(subdir)
        if final_count < total_in_zip:
            return (
                zip_path.name, "WARN", elapsed,
                f"only {final_count}/{total_in_zip} files after extraction"
            )

        return zip_path.name, status_prefix, elapsed, f"{final_count}/{total_in_zip} files"

    except Exception as e:
        return zip_path.name, "FAIL", time.time() - start, str(e)


def format_eta(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


def main():
    parser = argparse.ArgumentParser(description="Parallel zip extractor with resume support")
    parser.add_argument(
        "source_dirs",
        nargs="*",
        default=["/media/rayan/usb1/osv5m/images/train",
                 "/media/rayan/usb1/osv5m/images/cloud"],
        help="Directories containing zip files (default: train + cloud)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: same as each source dir)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 4; use 1-2 for slow USB HDDs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without extracting",
    )
    args = parser.parse_args()

    # Collect all (zip_path, output_dir) tasks across all source dirs
    all_tasks = []
    for src in args.source_dirs:
        source = Path(src)
        if not source.exists():
            print(f"WARNING: {source} does not exist, skipping.")
            continue
        output = Path(args.output_dir) if args.output_dir else source
        output.mkdir(parents=True, exist_ok=True)
        zips = sorted(source.glob("*.zip"))
        if not zips:
            print(f"WARNING: no zip files found in {source}")
            continue
        for z in zips:
            all_tasks.append((z, output, args.workers == 1))

    if not all_tasks:
        print("No zip files found in any source directory.")
        sys.exit(1)

    total_size = sum(z.stat().st_size for z, _, _ in all_tasks)
    print(f"Found {len(all_tasks)} zip files ({total_size / 1e9:.1f} GB total)")
    print(f"Workers    : {args.workers}")
    print()

    if args.dry_run:
        for z, out, _ in all_tasks:
            stem = z.stem
            subdir = out / stem
            already = count_extracted_files(subdir)
            total = count_zip_files(z)
            state = "DONE" if already >= total > 0 else ("PARTIAL" if already > 0 else "TODO")
            print(f"  [{state:<7}] {z.parent.name}/{z.name}  ({z.stat().st_size / 1e9:.2f} GB)  {already}/{total} files")
        return

    done = 0
    failed = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_zip, t): t[0] for t in all_tasks}
        for future in as_completed(futures):
            name, status, elapsed, info = future.result()
            done += 1
            elapsed_total = time.time() - t0
            rate = done / elapsed_total if elapsed_total > 0 else 1
            remaining = (len(all_tasks) - done) / rate if rate > 0 else 0

            time_str = f"{elapsed:5.1f}s" if elapsed > 0 else "  ---  "
            print(
                f"[{done:>3}/{len(all_tasks)}] {status:<7}  {name:<12}  "
                f"{time_str}  {info}  |  ETA {format_eta(remaining)}"
            )
            if status == "FAIL":
                failed.append((name, info))

    print()
    total_elapsed = time.time() - t0
    print(f"Done in {format_eta(total_elapsed)}  ({len(all_tasks) - len(failed)}/{len(all_tasks)} succeeded)")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err or 'unknown error'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
