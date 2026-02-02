from __future__ import annotations
import os
from common import (
    build_common_argparser, cfg_from_args, load_tweeteval_sentiment, tfidf_svd_angles,
    make_qnode_angle, run_and_save, DATASET_NAME,
    make_trainable_qnode_angle, run_and_save_vqc,
)


def main():
    p = build_common_argparser("(1) Angle encoding + Linear entanglement (baseline)")
    p.add_argument("--vqc", action="store_true", help="Use trainable VQC instead of fixed feature-map + LR.")
    p.add_argument("--vqc_steps", type=int, default=200)
    p.add_argument("--vqc_lr", type=float, default=0.05)
    p.add_argument("--vqc_batch", type=int, default=64)
    p.add_argument("--vqc_var_layers", type=int, default=1)
    p.add_argument("--vqc_class_weights", action="store_true", help="Enable class weighting in VQC loss")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    results_csv = os.path.join(outdir, "results_checklist_all_methods.csv")
    results_json = os.path.join(outdir, "results_checklist_all_methods.json")

    data = load_tweeteval_sentiment()
    prep = tfidf_svd_angles(
        data["X_text_train"], data["X_text_val"], data["X_text_test"],
        n_components=cfg.n_qubits,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        random_state=cfg.random_state,
        cache_dir=args.cache_dir,
    )

    if args.vqc:
        qnode = make_trainable_qnode_angle(
            n_qubits=cfg.n_qubits,
            rot_mode=cfg.rot_mode,
            axis=cfg.axis,
            entangler=cfg.entangler,
            topology=cfg.topology,
            ent_pattern="linear",
            L=1,
            readout_mode="Z",
            var_layers=args.vqc_var_layers,
            device_name=cfg.device_name,
            shots=cfg.shots,
        )

        row = run_and_save_vqc(
            technique_name="(1) angle + linear entanglement | readout=Z | VQC-trainable",
            qnode_trainable=qnode,
            X_train_in=prep["X_train_ang"],
            X_val_in=prep["X_val_ang"],
            X_test_in=prep["X_test_ang"],
            y_train=data["y_train"],
            y_val=data["y_val"],
            y_test=data["y_test"],
            q=cfg.n_qubits,
            L=1,
            entanglement_pattern_label=f"linear({cfg.topology},{cfg.entangler})+vqc",
            feature_dim=cfg.n_qubits,
            results_csv=results_csv,
            results_json=results_json,
            preproc_time_s=prep["preproc_time_s"],
            pattern_for_proxy="linear",
            topology=cfg.topology,
            rot_mode=cfg.rot_mode,
            block_size_for_proxy=2,
            var_layers=args.vqc_var_layers,
            steps=args.vqc_steps,
            lr=args.vqc_lr,
            batch_size=args.vqc_batch,
            seed=cfg.random_state,
            use_class_weights=args.vqc_class_weights
        )
    else:
        qnode = make_qnode_angle(
            n_qubits=cfg.n_qubits,
            rot_mode=cfg.rot_mode,
            axis=cfg.axis,
            entangler=cfg.entangler,
            topology=cfg.topology,
            ent_pattern="linear",
            L=1,
            readout_mode="Z",
            device_name=cfg.device_name,
            shots=cfg.shots,
        )

        row = run_and_save(
            technique_name="(1) angle + linear entanglement | readout=Z",
            qnode=qnode,
            X_train_in=prep["X_train_ang"],
            X_val_in=prep["X_val_ang"],
            X_test_in=prep["X_test_ang"],
            y_train=data["y_train"],
            y_val=data["y_val"],
            y_test=data["y_test"],
            q=cfg.n_qubits,
            L=1,
            entanglement_pattern_label=f"linear({cfg.topology},{cfg.entangler})",
            feature_dim=cfg.n_qubits,
            results_csv=results_csv,
            results_json=results_json,
            preproc_time_s=prep["preproc_time_s"],
            pattern_for_proxy="linear",
            topology=cfg.topology,
            rot_mode=cfg.rot_mode,
            block_size_for_proxy=2,
        )

    print("Saved:", row)

if __name__ == "__main__":
    main()
