from __future__ import annotations
import os
from common import (
    build_common_argparser, cfg_from_args, load_tweeteval_sentiment,
    hashing_amp_vectors, make_qnode_amplitude, run_and_save
)

def main():
    p = build_common_argparser("(5) Amplitude encoding (hashing -> state prep)")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    q = int(cfg.q_amp)
    k_amp = 2 ** q

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    results_csv = os.path.join(outdir, "results_checklist_all_methods.csv")
    results_json = os.path.join(outdir, "results_checklist_all_methods.json")

    data = load_tweeteval_sentiment()

    # Preprocess amplitude vectors (hashing + TF-IDF)
    import time
    t0 = time.perf_counter()
    Xtr_amp = hashing_amp_vectors(data["X_text_train"], k_amp=k_amp, random_state=cfg.random_state)
    Xva_amp = hashing_amp_vectors(data["X_text_val"],   k_amp=k_amp, random_state=cfg.random_state)
    Xte_amp = hashing_amp_vectors(data["X_text_test"],  k_amp=k_amp, random_state=cfg.random_state)
    preproc_time_s = time.perf_counter() - t0

    qnode = make_qnode_amplitude(n_qubits=q, device_name=cfg.device_name, shots=cfg.shots)

    row = run_and_save(
        technique_name=f"(5) amplitude encoding (hashing k=2^{q}={k_amp}) | readout=Z",
        qnode=qnode,
        X_train_in=Xtr_amp,
        X_val_in=Xva_amp,
        X_test_in=Xte_amp,
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=q,
        L=1,
        entanglement_pattern_label="stateprep(amplitude)",
        feature_dim=q,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=float(preproc_time_s),
        pattern_for_proxy="amplitude_stateprep",
        topology="chain",
        rot_mode="single",
        block_size_for_proxy=2,
        time_ref_X=Xtr_amp,
    )
    print("Saved:", row)

if __name__ == "__main__":
    main()
