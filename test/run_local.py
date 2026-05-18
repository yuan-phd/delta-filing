"""
Delta Filing — Run All Local Tests
No GPU needed. Run from project root:
    python tests/run_local.py
"""
import subprocess
import sys
import os

TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(TEST_DIR)

tests = [
    ("diff.py", "test_diff.py"),
    ("edgar.py", "test_edgar.py"),
    ("data quality", "test_data_quality.py"),
    ("app.py", "test_app.py"),
]

print("=" * 60)
print("  Delta Filing — Local Test Suite")
print("=" * 60)

total_pass = 0
total_fail = 0

for name, filename in tests:
    print(f"\n{'─'*60}")
    print(f"  Testing: {name}")
    print(f"{'─'*60}")

    result = subprocess.run(
        [sys.executable, os.path.join(TEST_DIR, filename)],
        cwd=PROJECT_DIR,
        capture_output=True, text=True,
    )

    print(result.stdout)
    if result.stderr:
        # Only print errors, not warnings
        errors = [l for l in result.stderr.split('\n') if 'Error' in l or 'error' in l]
        if errors:
            for e in errors:
                print(f"  {e}")

    # Parse pass/fail from output
    for line in result.stdout.split('\n'):
        if 'passed' in line:
            parts = line.strip().split()
            for p in parts:
                if '/' in p:
                    try:
                        passed, total = p.split('/')
                        total_pass += int(passed)
                        total_fail += int(total) - int(passed)
                    except ValueError:
                        pass

print(f"\n{'=' * 60}")
print(f"  ALL LOCAL TESTS: {total_pass}/{total_pass + total_fail} passed")
if total_fail > 0:
    print(f"  {total_fail} FAILURES")
else:
    print(f"  All tests passed!")
print(f"{'=' * 60}")
