"""Smoke test for a running API (local uvicorn, Docker or Kubernetes).

    python scripts/smoke_test.py                         # localhost:8000
    python scripts/smoke_test.py --url http://localhost:8080

Exits with a non-zero code if any check fails, so CI can use it directly.
Uses only the standard library, so it runs anywhere Python does.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def call(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def wait_until_up(base, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if call(f"{base}/health")[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=int, default=30,
                        help="seconds to wait for the service to come up")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    if not wait_until_up(base, args.wait):
        print(f"FAIL  {base}/health not ready after {args.wait}s")
        sys.exit(1)

    checks = []
    status, body = call(f"{base}/health")
    checks.append(("health is ok", status == 200 and body.get("model_loaded")))

    status, body = call(f"{base}/predict", PATIENT)
    print("Prediction:", json.dumps(body, indent=2))
    checks.append(("predict returns 200", status == 200))
    checks.append(("prediction is 0 or 1", body.get("prediction") in (0, 1)))
    checks.append(("confidence in [0.5, 1]",
                   0.5 <= body.get("confidence", -1) <= 1))

    status, _ = call(f"{base}/predict", {**PATIENT, "age": 5})
    checks.append(("invalid input rejected (422)", status == 422))

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
