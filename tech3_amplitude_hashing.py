"""
Tech3: Amplitude Encoding (Hashing)
"""
import argparse
import time
from common import (
    ExperimentConfig,
    load_tweeteval_sentiment,
    hashing_amp_vectors,
    make_qnode_amplitude,
    evaluate_technique,
    save_result,
)


def main():
    parser = argparse.ArgumentParser(description="Tech3: Amplitude Encoding (Hashing)")
    parser.add_argument("--q", type=int, default=6, help="Number of qubits (k=2^q)")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--clf", type=str, default="linearsvc", choices=["linearsvc", "ridge"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="results_all_techniques.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    k = 2 ** args.q

    # Configuration
    cfg = ExperimentConfig(
        random_state=args.seed,
        shots=args.shots,
        n_qubits=args.q,
        L=1,  # Amplitude typically uses L=1
        clf_type=args.clf,
        n_runs=args.n_runs,
    )

    print(f"\n{'#' * 60}")
    print(f"# Tech3: Amplitude Encoding (Hashing)")
    print(f"# q={args.q}, k=2^{args.q}={k}")
    print(f"{'#' * 60}")

    # Load data
    print("\nLoading TweetEval dataset...")
    data = load_tweeteval_sentiment()

    # Preprocess
    print(f"Preprocessing (Hashing to k={k} dimensions)...")
    t0 = time.time()
    X_train = hashing_amp_vectors(data["X_text_train"], k, cfg.random_state)
    X_val = hashing_amp_vectors(data["X_text_val"], k, cfg.random_state)
    X_test = hashing_amp_vectors(data["X_text_test"], k, cfg.random_state)
    preproc_time = time.time() - t0

    # Create quantum circuit
    print("Creating quantum circuit...")
    qnode = make_qnode_amplitude(
        args.q,
        L=1,
        entangler="none",
        device_name=cfg.device_name,
        shots=args.shots,
    )

    # Evaluate
    result = evaluate_technique(
        technique_name=f"Tech3_Amplitude_Hashing_q{args.q}_k{k}",
        qnode=qnode,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        preproc_time=preproc_time,
        cfg=cfg,
        extra_params={
            "entanglement_pattern": "none",
            "encoding": "amplitude",
            "k": k,
        },
    )

    # Save
    save_result(result, args.output)


if __name__ == "__main__":
    main()