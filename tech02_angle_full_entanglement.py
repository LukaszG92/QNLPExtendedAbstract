from __future__ import annotations
import os
from common import (
    build_common_argparser, cfg_from_args, load_tweeteval_sentiment, tfidf_svd_angles,
    make_qnode_angle, run_and_save
)

def main():
    p = build_common_argparser("(2) Angle encoding + Full entanglement (all-to-all)")
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

    qnode = make_qnode_angle(
        n_qubits=cfg.n_qubits,
        rot_mode=cfg.rot_mode,
        axis=cfg.axis,
        entangler=cfg.entangler,
        topology=cfg.topology,
        ent_pattern="all_to_all",
        L=1,
        readout_mode="Z",
        device_name=cfg.device_name,
        shots=cfg.shots,
    )

    row = run_and_save(
        technique_name="(2) angle + full entanglement (all-to-all) | readout=Z",
        qnode=qnode,
        X_train_in=prep["X_train_ang"],
        X_val_in=prep["X_val_ang"],
        X_test_in=prep["X_test_ang"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=cfg.n_qubits,
        L=1,
        entanglement_pattern_label=f"all_to_all({cfg.entangler})",
        feature_dim=cfg.n_qubits,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=prep["preproc_time_s"],
        pattern_for_proxy="all_to_all",
        topology=cfg.topology,
        rot_mode=cfg.rot_mode,
        block_size_for_proxy=2,
    )
    print("Saved:", row)

if __name__ == "__main__":
    main()
