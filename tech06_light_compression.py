from __future__ import annotations
import os
from common import (
    build_common_argparser, cfg_from_args, load_tweeteval_sentiment,
    light_compress_from_word_tfidf,
    make_qnode_angle, make_qnode_amplitude, run_and_save
)

def main():
    p = build_common_argparser("(6) Lightweight input compression (base -> proj to q / 2^q)")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    q_angle = int(cfg.n_qubits)
    q_amp = int(cfg.q_amp)
    k_amp = 2 ** q_amp
    k_base = int(cfg.k_base)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    results_csv = os.path.join(outdir, "results_checklist_all_methods.csv")
    results_json = os.path.join(outdir, "results_checklist_all_methods.json")

    data = load_tweeteval_sentiment()

    prep = light_compress_from_word_tfidf(
        data["X_text_train"], data["X_text_val"], data["X_text_test"],
        n_qubits=q_angle,
        k_amp=k_amp,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        random_state=cfg.random_state,
        k_base=k_base,
        cache_dir=args.cache_dir,
    )

    # (6a) base -> proj to q (angle)
    qnode_angle = make_qnode_angle(
        n_qubits=q_angle,
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
    row_a = run_and_save(
        technique_name=f"(6a) light-compress (k_base={prep['k_base']} -> q={q_angle}) + angle + linear entanglement | readout=Z",
        qnode=qnode_angle,
        X_train_in=prep["X_train_lc_ang"],
        X_val_in=prep["X_val_lc_ang"],
        X_test_in=prep["X_test_lc_ang"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=q_angle,
        L=1,
        entanglement_pattern_label=f"linear({cfg.topology},{cfg.entangler})",
        feature_dim=q_angle,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=prep["preproc_time_s"],
        pattern_for_proxy="linear",
        topology=cfg.topology,
        rot_mode=cfg.rot_mode,
        block_size_for_proxy=2,
    )

    # (6b) base -> proj to 2^q (amplitude)
    qnode_amp = make_qnode_amplitude(n_qubits=q_amp, device_name=cfg.device_name, shots=cfg.shots)
    row_b = run_and_save(
        technique_name=f"(6b) light-compress (k_base={prep['k_base']} -> 2^q={k_amp}) + amplitude stateprep | readout=Z",
        qnode=qnode_amp,
        X_train_in=prep["X_train_amp_lc"],
        X_val_in=prep["X_val_amp_lc"],
        X_test_in=prep["X_test_amp_lc"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=q_amp,
        L=1,
        entanglement_pattern_label="stateprep(amplitude)",
        feature_dim=q_amp,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=prep["preproc_time_s"],
        pattern_for_proxy="amplitude_stateprep",
        topology="chain",
        rot_mode="single",
        block_size_for_proxy=2,
        time_ref_X=prep["X_train_amp_lc"],
    )

    print("Saved:", row_a)
    print("Saved:", row_b)

if __name__ == "__main__":
    main()
