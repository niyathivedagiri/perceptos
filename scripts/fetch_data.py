#!/usr/bin/env python3
"""
PerceptOS dataset fetcher.

The repository ships the *recipe*, never the bytes. Everything here is public
NASA, JPL or USGS data; nothing is redistributed.

    python scripts/fetch_data.py list
    python scripts/fetch_data.py perseverance --sol 203 --camera NAVCAM --limit 6
    python scripts/fetch_data.py jezero-dem
    python scripts/fetch_data.py ai4mars
    python scripts/fetch_data.py status

Get a free NASA API key at https://api.nasa.gov and export it:

    export NASA_API_KEY=your_key_here

DEMO_KEY works but is rate-limited to roughly 30 requests per hour by IP,
which is not enough for a bulk fetch.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.data.cache import DatasetCache, HttpError, RateLimited  # noqa: E402
from perceptos.data.mars2020 import Mars2020Client  # noqa: E402
from perceptos.data.mars2020_raw import JplRawImageClient  # noqa: E402
from perceptos.data.sources import (  # noqa: E402
    NASA_SERVICES,
    credentials_report,
)
from perceptos.core.environment import ENVIRONMENTS, get_environment  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "datasets"


# --------------------------------------------------------------------------
# Large manual downloads: these are too big to pull unattended, so the tool
# prints instructions rather than silently starting an 18 GB transfer.
# --------------------------------------------------------------------------

MANUAL_DATASETS = {
    "ai4mars": {
        "size": "17.8 GB",
        "url": "https://zenodo.org/records/15995036",
        "file": "ai4mars-dataset-merged-0.6.zip",
        "target": "ai4mars",
        "notes": textwrap.dedent("""\
            AI4Mars v0.6: ~425K semantic segmentation labels over ~50K images
            from Perseverance, Curiosity, Opportunity and Spirit.

            Label rasters are greyscale; pixel values are class ids:
                0 = soil   1 = bedrock   2 = sand   3 = big rock
                255 = unlabelled  (NOT a class - do not train on it)

            Cite: Swan, R. M., et al. AI4MARS: A Dataset for Terrain-Aware
            Autonomous Driving on Mars. CVPRW 2021.

            The full download is large. For development, extract only the
            msl/images/edr + msl/labels/train subset (~2 GB).
        """),
    },
    "jezero-dem": {
        "size": "~1-3 GB per product",
        "url": ("https://astrogeology.usgs.gov/search/map/Mars/Mars2020/"
                "JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40"),
        "file": "Jezero HiRISE DTM, 1 m/pixel",
        "target": "jezero_dem",
        "notes": textwrap.dedent("""\
            HiRISE digital terrain model of Jezero crater at 1 m/pixel,
            controlled to MOLA. This is the terrain Perseverance actually
            navigates.

            Alternatives:
              - Cloud Optimized GeoTIFFs (ranged reads, no full download):
                https://registry.opendata.aws/nasa-usgs-mars-hirise-dtms/
              - MSR extension mosaics (13.2 GB), covering the western rim:
                https://astrogeology.usgs.gov/search/map/
                mars-sample-return-terrain-relative-navigation-hirise-dtm-mosaic
              - CTX DEM mosaic at 20 m/pixel for a much smaller download

            IMPORTANT after download: no-data is a large negative sentinel,
            not NaN. perceptos.data.terrain.load_dem converts it; loading the
            raster yourself without that step puts a -3.4e38 m cliff in your
            terrain, which the hazard analyser will faithfully report as the
            most dangerous surface in the solar system.

            To use it, crop a tile and save as .npy:
                import rasterio, numpy as np
                with rasterio.open("DTEEC_....IMG") as ds:
                    np.save("datasets/jezero_dem/tile.npy", ds.read(1)[5000:5600, 5000:5600])
        """),
    },
}


def cmd_credentials(args: argparse.Namespace) -> int:
    """Answer 'do I need a NASA API key?' from the code, not from memory."""
    report = credentials_report()
    print(f"\nNASA credentials  (verified {report['verified_utc']})")
    print("=" * 74)
    print(f"  {report['summary']}\n")
    print(f"  api.nasa.gov key status : {report['api_nasa_gov_key_status']}")
    print(f"  environment variable    : {report['env_var']}")
    print(f"  primary source          : {report['primary_perseverance_source']} "
          f"(key required: {report['primary_source_requires_key']})\n")
    for service in NASA_SERVICES.values():
        mark = "KEY REQUIRED" if service.requires_api_key else "no key needed"
        print(f"  [{mark}] {service.name}")
        print(f"      endpoint   : {service.endpoint}")
        print(f"      provides   : {service.provides[:100]}")
        print(f"      rate limit : {service.rate_limit}")
        print(f"      env var    : {service.env_var or '-'}")
        print(f"      used by    : {service.used_by}")
        print(f"      if missing : {service.on_missing_key}")
        print(f"      credit     : {service.credit}\n")
    return 0


def cmd_perseverance_jpl(args: argparse.Namespace) -> int:
    """Fetch from the keyless JPL archive — the preferred Perseverance source."""
    cache = DatasetCache(args.root)
    client = JplRawImageClient(cache)
    print("Using the JPL Mars 2020 Raw Image API. No API key is sent, because "
          "none is required.\n")

    total = 0
    for sol in _parse_sols(args.sol):
        print(f"sol {sol}: querying {args.camera}...")
        try:
            left = client.query(sol=sol, instrument=f"{args.camera}_LEFT", num=args.num)
            right = client.query(sol=sol, instrument=f"{args.camera}_RIGHT", num=args.num)
        except HttpError as exc:
            print(f"  failed: {exc}")
            continue

        images = left + right
        if not images:
            print("  no images returned for this sol/camera")
            continue
        with_cal = sum(1 for i in images if i.has_calibration)
        print(f"  {len(images)} image(s), {with_cal} with a published camera model")
        if client.last_warnings:
            for warning in client.last_warnings[:3]:
                print(f"  note: {warning}")

        pairs = client.find_stereo_pairs(images)
        print(f"  {len(pairs)} candidate stereo pair(s)")
        for pair in pairs[: args.limit]:
            print(f"    sol {pair.sol}  {pair.status.value}  "
                  f"conf {pair.confidence:.2f}  "
                  f"baseline {pair.baseline_m if pair.baseline_m else 'UNKNOWN'}")
            for concern in pair.concerns[:2]:
                print(f"      ! {concern}")
            if client.download_pair(pair):
                total += 1

        for image in images[: args.limit * 2]:
            if image.local_path is None:
                try:
                    client.download(image)
                except HttpError:
                    pass

    print(f"\n{total} verified pair(s) cached under "
          f"{cache.dataset_dir(client.dataset)}")
    print("Pairs are RAW. Depth is only offered for RECTIFIED_STEREO; nothing "
          "below that status gets a fabricated baseline.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("\nAutomatic fetchers")
    print("-" * 70)
    print(f"  {'perseverance':<16} JPL raw archive (NO API KEY) - preferred")
    print(f"  {'perseverance-api':<16} api.nasa.gov Mars Rover Photos (needs a key)")
    print("\nManual downloads (too large to pull unattended)")
    print("-" * 70)
    for name, spec in MANUAL_DATASETS.items():
        print(f"  {name:<16} {spec['size']:<14} {spec['url'][:44]}...")
    print("\nEnvironments and their data sources")
    print("-" * 70)
    for name, env in ENVIRONMENTS.items():
        payload = env.to_dict()["real_data_available"]
        flags = "".join(
            f" {k}={'yes' if v else 'no':<3}" for k, v in payload.items()
        )
        print(f"  {name:<12}{flags}")
    print()
    return 0


def cmd_perseverance(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
    if api_key == "DEMO_KEY":
        print("! Using DEMO_KEY (about 30 requests/hour). Get a free key at "
              "https://api.nasa.gov\n")

    cache = DatasetCache(args.root)
    client = Mars2020Client(cache, api_key=api_key)

    sols = _parse_sols(args.sol)
    total_pairs = 0
    for sol in sols:
        print(f"sol {sol}: querying {args.camera}...")
        try:
            pairs = client.stereo_pairs(sol, family=args.camera)
        except RateLimited:
            print("  rate limited. Wait an hour, or use a real API key.")
            return 2
        except HttpError as exc:
            print(f"  failed: {exc}")
            continue

        if not pairs:
            print("  no stereo pairs found for this sol/camera")
            continue

        print(f"  {len(pairs)} stereo pair(s) matched on spacecraft clock")
        downloaded = client.download_pairs(pairs, limit=args.limit)
        total_pairs += len(downloaded)
        for pair in downloaded:
            print(f"    sol {pair.sol} sclk {pair.sclk} "
                  f"{pair.camera_family} -> {pair.left.local_path.parent}")

    print(f"\n{total_pairs} pair(s) cached under {cache.dataset_dir(client.dataset)}")
    print("Pairs are RAW (unrectified). StereoDatasetSource will refuse to hand "
          "them to a block matcher until they are rectified.")
    return 0


def cmd_manual(args: argparse.Namespace) -> int:
    spec = MANUAL_DATASETS[args.command]
    target = Path(args.root) / spec["target"]
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n{args.command}  ({spec['size']})")
    print("=" * 70)
    print(f"Download from: {spec['url']}")
    print(f"File:          {spec['file']}")
    print(f"Extract to:    {target}\n")
    print(spec["notes"])
    doc = target / "SOURCE.md"
    if not doc.exists():
        doc.write_text(
            f"# Data source — {args.command}\n\n"
            f"- **URL:** {spec['url']}\n"
            f"- **Size:** {spec['size']}\n"
            f"- **Credit:** NASA/JPL-Caltech (and USGS for terrain products)\n\n"
            f"{spec['notes']}\n"
        )
        print(f"Wrote {doc}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    stats = DatasetCache(args.root).statistics()
    print(f"\nCache root: {stats['root']}")
    if not stats["datasets"]:
        print("  (empty — nothing fetched yet)")
        print("\nPerceptOS runs in SIMULATION mode until real data is present.")
        return 0
    total = 0
    for name, info in stats["datasets"].items():
        total += info["bytes"]
        print(f"  {name:<24} {info['files']:>6} files  {info['bytes'] / 1e6:>9.1f} MB")
    print(f"  {'TOTAL':<24} {'':>6}        {total / 1e6:>9.1f} MB\n")
    return 0


def _parse_sols(value: str) -> list[int]:
    """Accept ``203`` or ``200-210`` or ``200,203,207``."""
    sols: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            sols.extend(range(int(start), int(end) + 1))
        elif part:
            sols.append(int(part))
    return sols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch public NASA/JPL/USGS datasets for PerceptOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="cache root (default: datasets/)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="show available datasets")
    subparsers.add_parser("status", help="show what is already cached")

    subparsers.add_parser("credentials",
                          help="which NASA services need a key (verified)")

    jpl = subparsers.add_parser(
        "perseverance", help="fetch from the keyless JPL raw archive (preferred)")
    jpl.add_argument("--sol", default="203", help="sol, range (200-210) or list")
    jpl.add_argument("--camera", default="NAVCAM",
                     choices=["NAVCAM", "FRONT_HAZCAM", "REAR_HAZCAM", "MCZ"])
    jpl.add_argument("--limit", type=int, default=8, help="max pairs per sol")
    jpl.add_argument("--num", type=int, default=100, help="feed page size")

    persy = subparsers.add_parser(
        "perseverance-api", help="fetch via api.nasa.gov (requires a key)")
    persy.add_argument("--sol", default="203", help="sol, range (200-210) or list")
    persy.add_argument("--camera", default="NAVCAM",
                       choices=["NAVCAM", "FRONT_HAZCAM", "REAR_HAZCAM", "MCZ"])
    persy.add_argument("--limit", type=int, default=8, help="max pairs per sol")
    persy.add_argument("--api-key", default=None)

    for name in MANUAL_DATASETS:
        subparsers.add_parser(name, help=f"instructions for {name}")

    args = parser.parse_args()
    if args.command == "list":
        return cmd_list(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "credentials":
        return cmd_credentials(args)
    if args.command == "perseverance":
        return cmd_perseverance_jpl(args)
    if args.command == "perseverance-api":
        return cmd_perseverance(args)
    if args.command in MANUAL_DATASETS:
        return cmd_manual(args)
    parser.error(f"unknown command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
