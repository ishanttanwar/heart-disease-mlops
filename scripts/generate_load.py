"""Send realistic traffic to the API so the dashboards have data to show.

    python scripts/generate_load.py                 # 200 requests
    python scripts/generate_load.py --requests 500 --delay 0.05

Patients are sampled from the cleaned dataset when it is available, so the
prediction mix looks like real traffic rather than one repeated record. A
small share of deliberately invalid requests is included so the error-rate
panel is not permanently flat.
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import SAMPLE_PATIENT  # noqa: E402

INVALID_SHARE = 0.08  # ~8% of requests are rejected on purpose


def load_patients():
    """Patients from the cleaned CSV, or jittered copies of the sample."""
    path = Path(config.CLEAN_DATA_PATH)
    if not path.exists():
        return [
            {**SAMPLE_PATIENT,
             "age": SAMPLE_PATIENT["age"] + random.randint(-15, 15),
             "thalach": SAMPLE_PATIENT["thalach"] + random.randint(-30, 30)}
            for _ in range(50)
        ]
    import pandas as pd
    df = pd.read_csv(path)[config.FEATURES]

    def clean(key, value):
        # ca and thal may be missing; the API accepts null for both
        if value is None or pd.isna(value):
            return None
        return float(value) if key in config.NUMERIC_FEATURES else int(value)

    return [
        {k: clean(k, v) for k, v in record.items()}
        for record in df.to_dict("records")
    ]


def post(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, None
    except Exception:
        return 0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.1,
                        help="seconds between requests")
    args = parser.parse_args()
    endpoint = args.url.rstrip("/") + "/predict"

    patients = load_patients()
    counts = {"disease": 0, "no_disease": 0, "rejected": 0, "failed": 0}
    start = time.perf_counter()

    for i in range(1, args.requests + 1):
        patient = dict(random.choice(patients))
        if random.random() < INVALID_SHARE:
            patient["age"] = random.choice([5, 150])   # outside the schema
        status, body = post(endpoint, patient)
        if status == 200:
            key = "disease" if body["prediction"] == 1 else "no_disease"
            counts[key] += 1
        elif status == 422:
            counts["rejected"] += 1
        else:
            counts["failed"] += 1
        if i % 25 == 0:
            print(f"{i}/{args.requests} sent", flush=True)
        time.sleep(args.delay)

    elapsed = time.perf_counter() - start
    print(f"\nSent {args.requests} requests in {elapsed:.1f}s "
          f"({args.requests / elapsed:.1f} req/s)")
    for name, value in counts.items():
        print(f"  {name:<12} {value}")
    if counts["failed"]:
        print("\nSome requests did not reach the API - is it running?")
        sys.exit(1)


if __name__ == "__main__":
    main()
