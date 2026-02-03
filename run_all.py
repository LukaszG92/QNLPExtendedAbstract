"""
Run ALL QNLP experiments in parallel.
Simple script - just run it and wait for results.
"""
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def run_experiment(script, q, L=None, shots=256, n_runs=5):
    """Run a single experiment"""
    cmd = [sys.executable, script, "--q", str(q), "--shots", str(shots), "--n_runs", str(n_runs)]
    if L is not None:
        cmd.extend(["--L", str(L)])

    name = f"{script.replace('.py', '')} q={q}" + (f" L={L}" if L else "")
    print(f"[START] {name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"[✓ DONE] {name}")
            return True
        else:
            print(f"[✗ FAIL] {name}")
            return False
    except Exception as e:
        print(f"[✗ ERROR] {name}: {e}")
        return False


def main():
    """Run all experiments"""
    print("\n" + "=" * 70)
    print("🚀 RUNNING ALL QNLP EXPERIMENTS")
    print("=" * 70)

    experiments = []
    workers = 12

    # Tech1: Angle + Linear (q=[4,6,8], L=[1,2,3])
    print("\n📋 Tech1: Angle + Linear")
    for q in [4, 6, 8]:
        for L in [1, 2, 3]:
            experiments.append(("tech1_angle_linear.py", q, L))

    # Tech2: Angle + All-to-All (q=[4,6,8], L=[1,2])
    print("📋 Tech2: Angle + All-to-All")
    for q in [4, 6, 8]:
        for L in [1, 2]:
            experiments.append(("tech2_angle_all_to_all.py", q, L))

    # Tech3: Amplitude + Hashing (q=[4,5,6])
    print("📋 Tech3: Amplitude + Hashing")
    for q in [4, 5, 6]:
        experiments.append(("tech3_amplitude_hashing.py", q, None))

    # Tech4: Word2Ket Angle (q=[4,6,8], L=[1,2])
    print("📋 Tech4: Word2Ket Angle")
    for q in [4, 6, 8]:
        for L in [1, 2]:
            experiments.append(("tech4_word2ket_angle.py", q, L))

    # Tech4: Word2Ket Amplitude (q=[4,5,6])
    print("📋 Tech4: Word2Ket Amplitude")
    for q in [4, 5, 6]:
        experiments.append(("tech4_word2ket_amplitude.py", q, None))

    print(f"\n📊 Total experiments: {len(experiments)}")
    print(f"⚙️  Workers: {workers} (parallel)")
    print(f"⏱️  Estimated time: ~25-35 minutes")
    print("=" * 70)

    response = input(f"\n▶️  Run {len(experiments)} experiments? (y/n): ")
    if response.lower() != 'y':
        print("❌ Aborted.")
        return

    print("\n🚀 Starting...\n")
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for script, q, L in experiments:
            future = executor.submit(run_experiment, script, q, L)
            futures.append(future)

        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.time() - start_time
    success_count = sum(results)

    # Summary
    print("\n" + "=" * 70)
    print("✅ COMPLETED!")
    print("=" * 70)
    print(f"✓ Success: {success_count}/{len(results)}")
    print(f"✗ Failed: {len(results) - success_count}/{len(results)}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"📁 Results saved in: results_all_techniques.csv")
    print("=" * 70)
    print("\n📊 Next step:")
    print("   python analyze_results.py")
    print()


if __name__ == "__main__":
    main()