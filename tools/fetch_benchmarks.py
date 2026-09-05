#!/usr/bin/env python3
"""Download the MAPF benchmark maps and scenarios, and verify them.

The benchmark set is not committed to this repository. It is ~19 MB of
third-party data with its own terms, and vendoring it into an MIT-licensed
package would be both rude and pointless when the upstream host is stable and
the archives are content-addressed by the SHA-256 values recorded in
``swarmplan.datasets``.

Usage::

    python3 tools/fetch_benchmarks.py --list
    python3 tools/fetch_benchmarks.py --all
    python3 tools/fetch_benchmarks.py maps scen-random
    python3 tools/fetch_benchmarks.py --verify
    python3 tools/fetch_benchmarks.py --describe

Files land in ``data/maps``, ``data/scen-random`` and ``data/scen-even``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarmplan import datasets  # noqa: E402
from swarmplan.graph import GridMap  # noqa: E402


def human(n: int) -> str:
    """Format a byte count."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, timeout: float = 300.0) -> None:
    """Download to a temporary file and move it into place when complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dest)


def extract(archive: Path, target: Path, suffix: str) -> int:
    """Extract every ``suffix`` member of a zip into ``target``, flattened.

    The upstream archives nest their contents one directory deep and the depth
    has changed before, so members are matched by extension and written by base
    name rather than by trusting the archive layout.
    """
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(suffix):
                continue
            name = Path(member).name
            if not name or name.startswith("."):
                continue
            with zf.open(member) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def fetch(keys, data_dir: Path, force: bool = False) -> int:
    """Fetch, verify and extract the named archives."""
    status = 0
    for key in keys:
        spec = datasets.ARCHIVES[key]
        archive = data_dir / spec.filename
        print(f"{spec.key}: {spec.description}")
        if archive.exists() and not force:
            print(f"  archive already present ({human(archive.stat().st_size)})")
        else:
            try:
                download(spec.url, archive)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  download failed: {exc}", file=sys.stderr)
                status = 1
                continue
        size = archive.stat().st_size
        digest = sha256_of(archive)
        if size != spec.size_bytes:
            print(f"  size mismatch: expected {spec.size_bytes}, got {size}", file=sys.stderr)
            status = 1
            continue
        if digest != spec.sha256:
            print(
                f"  SHA-256 mismatch:\n    expected {spec.sha256}\n    got      {digest}",
                file=sys.stderr,
            )
            status = 1
            continue
        print(f"  verified {human(size)}  sha256 {digest[:16]}...")
        suffix = ".map" if spec.key == "maps" else ".scen"
        n = extract(archive, data_dir / spec.extract_to, suffix)
        print(f"  extracted {n} {suffix} files into data/{spec.extract_to}/")
    return status


def verify(data_dir: Path) -> int:
    """Re-check the archives on disk against the recorded hashes."""
    status = 0
    for spec in datasets.ARCHIVES.values():
        archive = data_dir / spec.filename
        if not archive.exists():
            print(f"{spec.key:12s} missing")
            status = 1
            continue
        digest = sha256_of(archive)
        ok = digest == spec.sha256 and archive.stat().st_size == spec.size_bytes
        print(f"{spec.key:12s} {'ok' if ok else 'MISMATCH'}  {digest}")
        status |= 0 if ok else 1
    return status


def describe(data_dir: Path) -> int:
    """Print the dimensions of every map we run on, read from the files."""
    print(f"{'map':26s} {'size':>10s} {'free':>7s} {'expected free':>14s}  note")
    status = 0
    for info in datasets.MAPS:
        path = datasets.map_path(info.name, data_dir)
        if not path.exists():
            print(f"{info.name:26s} {'missing':>10s}")
            status = 1
            continue
        grid = GridMap.from_file(path)
        flag = "" if grid.n_free == info.free_cells else "  <-- MISMATCH"
        status |= 0 if grid.n_free == info.free_cells else 1
        print(
            f"{info.name:26s} {grid.height:4d}x{grid.width:<5d} {grid.n_free:7d} "
            f"{info.free_cells:14d}{flag}  {info.why}"
        )
    return status


def main(argv=None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("keys", nargs="*", help="archives to fetch (default: all)")
    p.add_argument("--all", action="store_true", help="fetch every archive")
    p.add_argument("--list", action="store_true", help="list the archives and exit")
    p.add_argument("--verify", action="store_true", help="re-check hashes of downloaded archives")
    p.add_argument("--describe", action="store_true", help="print map dimensions read from disk")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()

    if args.list:
        for spec in datasets.ARCHIVES.values():
            print(f"{spec.key:12s} {human(spec.size_bytes):>10s}  {spec.url}")
            print(f"{'':12s} {'':>10s}  sha256 {spec.sha256}")
            print(f"{'':12s} {'':>10s}  {spec.description}")
        return 0
    if args.verify:
        return verify(data_dir)
    if args.describe:
        return describe(data_dir)

    unknown = [k for k in args.keys if k not in datasets.ARCHIVES]
    if unknown:
        print(f"unknown archive {unknown}; expected {list(datasets.ARCHIVES)}", file=sys.stderr)
        return 2
    keys = list(datasets.ARCHIVES) if args.all or not args.keys else args.keys
    return fetch(keys, data_dir, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
