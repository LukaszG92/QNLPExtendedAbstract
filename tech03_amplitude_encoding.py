from __future__ import annotations

import os
import time

from common import (
    build_common_argparser,
    cfg_from_args,
    load_tweeteval_sentiment,
    hashing_amp_vectors,
    make_qnode_amplitude,
    run_and_save,
    extra_fields_from_args,
)


def main():
    p = build_common_argparser("(3) Amplitude encoding (hashing -> state prep)")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    if os.environ.get("QNLP_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on"):
        print("[DEBUG] tech03_amplitude_encoding: args=", vars(args))
        print("[DEBUG] tech03_amplitude_encoding: cfg=", cfg)

    # Prefer explicit q_amp if present in cfg, otherwise fall back to n_qubits.
    q = int(getattr(cfg, "q_amp", getattr(cfg, "n_qubits", 0)))
    if q <= 0:
        raise ValueError(
            "Invalid number of qubits for amplitude encoding. "
            "Expected cfg.q_amp (or cfg.n_qubits) to be a positive integer."
        )
    k_amp = 2 ** q

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    results_csv = os.path.join(outdir, "results_checklist_all_methods.csv")
    results_json = os.path.join(outdir, "results_checklist_all_methods.json")

    data = load_tweeteval_sentiment()

    # Preprocess amplitude vectors: hashing -> (optionally TF/IDF inside helper) -> L2-normalized length k_amp
    t0 = time.perf_counter()
    Xtr_amp = hashing_amp_vectors(
        data["X_text_train"], k_amp=k_amp, random_state=cfg.random_state
    )
    Xva_amp = hashing_amp_vectors(
        data["X_text_val"], k_amp=k_amp, random_state=cfg.random_state
    )
    Xte_amp = hashing_amp_vectors(
        data["X_text_test"], k_amp=k_amp, random_state=cfg.random_state
    )
    preproc_time_s = time.perf_counter() - t0

    if os.environ.get("QNLP_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on"):
        print(
            "[DEBUG] prep shapes:",
            "X_train_amp",
            getattr(Xtr_amp, "shape", None),
            "X_val_amp",
            getattr(Xva_amp, "shape", None),
            "X_test_amp",
            getattr(Xte_amp, "shape", None),
        )

    qnode = make_qnode_amplitude(
        n_qubits=q,
        L=int(getattr(cfg, 'L', 1)),
        entangler=str(getattr(cfg, 'entangler', 'none')),
        topology=str(getattr(cfg, 'topology', 'chain')),
        device_name=cfg.device_name,
        shots=cfg.shots,
    )

    extra = extra_fields_from_args(args)
    clf_kind = str(extra.get("clf_kind", "linearsvc")).lower()

    row = run_and_save(
        technique_name=f"(3) amplitude encoding (hashing k=2^{q}={k_amp}) | q={q} | clf={clf_kind}",
        qnode=qnode,
        X_train_in=Xtr_amp,
        X_val_in=Xva_amp,
        X_test_in=Xte_amp,
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=q,
        L=int(getattr(cfg, 'L', 1)),
        entanglement_pattern_label=(
            f"stateprep(amplitude)" if int(getattr(cfg, 'L', 1)) == 1 and str(getattr(cfg, 'entangler', 'none')) == 'none'
            else f"amplitude-reupload(L={int(getattr(cfg,'L',1))})|{str(getattr(cfg,'entangler','none'))}|{str(getattr(cfg,'topology','chain'))}"
        ),
        feature_dim=q,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=float(preproc_time_s),
        pattern_for_proxy="amplitude_stateprep",
        # Keep these fields stable for the proxy/depth/#2q-gates estimation logic.
        topology="chain",
        rot_mode="single",
        block_size_for_proxy=2,
        time_ref_X=Xtr_amp,
        extra_fields=extra,
    )

    print("Saved:", row)


if __name__ == "__main__":
    main()
