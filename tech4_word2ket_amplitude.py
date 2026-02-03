"""
Tech4: Word2Ket Amplitude Encoding
"""
import argparse
import time
from common import (
    ExperimentConfig,
    load_tweeteval_sentiment,
    word2ket_embed_texts,
    make_qnode_amplitude,
    evaluate_technique,
    save_result,
)


def main():
    parser = argparse.ArgumentParser(description="Tech4: Word2Ket Amplitude Encoding")
    parser.add_argument("--q", type=int, default=6, help="Number of qubits (k=2^q)")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--entangler", type=str, default="none", choices=["cnot", "cz", "none"])
    parser.add_argument("--topology", type=str, default="chain", choices=["chain", "ring"])
    parser.add_argument("--clf", type=str, default="linearsvc", choices=["linearsvc", "ridge"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="results_all_techniques.csv")
    parser.add_argument("--seed", type=int, default=42)
    # Word2Ket specific
    parser.add_argument("--w2k_order", type=int, default=4)
    parser.add_argument("--w2k_rank", type=int, default=1)
    args = parser.parse_args()

    k = 2 ** args.q

    # Configuration
    cfg = ExperimentConfig(
        random_state=args.seed,
        shots=args.shots,
        n_qubits=args.q,
        entangler=args.entangler,
        topology=args.topology,
        L=1,  # Amplitude typically uses L=1
        clf_type=args.clf,
        n_runs=args.n_runs,
    )

    print(f"\n{'#'*60}")
    print(f"# Tech4: Word2Ket Amplitude Encoding")
    print(f"# q={args.q}, k=2^{args.q}={k}, entangler={args.entangler}")
    print(f"{'#'*60}")

    # Load data
    print("\nLoading TweetEval dataset...")
    data = load_tweeteval_sentiment()

    # Preprocess with Word2Ket
    print("Preprocessing (Word2Ket embeddings)...")
    w2k_cfg = {
        "embedding_dim": k,  # For amplitude, embedding_dim = 2^q
        "order": args.w2k_order,
        "rank": args.w2k_rank,
        "use_xs": True,
    }

    t0 = time.time()
    X_train = word2ket_embed_texts(
        list(data["X_text_train"]), "amplitude", args.q, w2k_cfg, cfg.random_state
    )
    X_val = word2ket_embed_texts(
        list(data["X_text_val"]), "amplitude", args.q, w2k_cfg, cfg.random_state
    )
    X_test = word2ket_embed_texts(
        list(data["X_text_test"]), "amplitude", args.q, w2k_cfg, cfg.random_state
    )
    preproc_time = time.time() - t0

    # Create quantum circuit
    print("Creating quantum circuit...")
    qnode = make_qnode_amplitude(
        args.q,
        L=1,
        entangler=args.entangler,
        topology=args.topology,
        device_name=cfg.device_name,
        shots=args.shots,
    )

    # Evaluate
    result = evaluate_technique(
        technique_name=f"Tech4_Word2Ket_Amplitude_q{args.q}_k{k}",
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
            "entanglement_pattern": f"linear_{args.entangler}" if args.entangler != "none" else "none",
            "encoding": "word2ket_amplitude",
            "k": k,
            "w2k_order": args.w2k_order,
            "w2k_rank": args.w2k_rank,
        },
    )

    # Save
    save_result(result, args.output)


if __name__ == "__main__":
    main()