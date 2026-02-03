"""
Tech1: Angle Encoding + Linear Entanglement
"""
import argparse
from common import (
    ExperimentConfig,
    load_tweeteval_sentiment,
    tfidf_svd_angles,
    make_qnode_angle,
    evaluate_technique,
    save_result,
)


def main():
    parser = argparse.ArgumentParser(description="Tech1: Angle + Linear Entanglement")
    parser.add_argument("--q", type=int, default=8, help="Number of qubits")
    parser.add_argument("--L", type=int, default=1, help="Number of layers")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--entangler", type=str, default="cnot", choices=["cnot", "cz"])
    parser.add_argument("--topology", type=str, default="chain", choices=["chain", "ring"])
    parser.add_argument("--axis", type=str, default="ry", choices=["rx", "ry", "rz"])
    parser.add_argument("--clf", type=str, default="linearsvc", choices=["linearsvc", "ridge"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="results_all_techniques.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Configuration
    cfg = ExperimentConfig(
        random_state=args.seed,
        shots=args.shots,
        n_qubits=args.q,
        entangler=args.entangler,
        topology=args.topology,
        axis=args.axis,
        L=args.L,
        clf_type=args.clf,
        n_runs=args.n_runs,
    )

    print(f"\n{'#' * 60}")
    print(f"# Tech1: Angle + Linear Entanglement")
    print(f"# q={args.q}, L={args.L}, entangler={args.entangler}, topology={args.topology}")
    print(f"{'#' * 60}")

    # Load data
    print("\nLoading TweetEval dataset...")
    data = load_tweeteval_sentiment()

    # Preprocess
    print("Preprocessing (TF-IDF + SVD)...")
    prep = tfidf_svd_angles(
        data["X_text_train"],
        data["X_text_val"],
        data["X_text_test"],
        n_components=args.q,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        random_state=cfg.random_state,
    )

    # Create quantum circuit
    print("Creating quantum circuit...")
    qnode = make_qnode_angle(
        args.q,
        entanglement="linear",
        entangler=args.entangler,
        topology=args.topology,
        axis=args.axis,
        L=args.L,
        device_name=cfg.device_name,
        shots=args.shots,
    )

    # Evaluate
    result = evaluate_technique(
        technique_name=f"Tech1_Angle_Linear_q{args.q}_L{args.L}",
        qnode=qnode,
        X_train=prep["X_train"],
        X_val=prep["X_val"],
        X_test=prep["X_test"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        preproc_time=prep["preproc_time"],
        cfg=cfg,
        extra_params={
            "entanglement_pattern": f"linear_{args.entangler}_{args.topology}",
            "encoding": "angle",
            "axis": args.axis,
        },
    )

    # Save
    save_result(result, args.output)


if __name__ == "__main__":
    main()