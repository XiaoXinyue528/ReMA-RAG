import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected", type=int, default=1127)
    parser.add_argument("--show-missing", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    files = sorted(out_dir.glob("plantucd_test_*.json"))
    done = set()
    bad = []

    for path in files:
        try:
            idx = int(path.stem.rsplit("_", 1)[-1])
            json.loads(path.read_text(encoding="utf-8"))
            done.add(idx)
        except Exception as exc:
            bad.append((path.name, str(exc)))

    missing = [i for i in range(args.expected) if i not in done]
    print(f"Output dir: {out_dir}")
    print(f"Done valid files: {len(done)}/{args.expected}")
    print(f"Bad json files: {len(bad)}")
    if missing:
        print(f"Missing first {min(args.show_missing, len(missing))}: {missing[:args.show_missing]}")
    if bad:
        print("Bad files:")
        for name, err in bad[:args.show_missing]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
