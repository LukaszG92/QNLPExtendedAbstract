from __future__ import annotations

import argparse
import json
import os
import time

import common


def _feature_dim_from_readout(q: int, readout_mode: str, include_ring_pair: bool) -> int:
    q = int(q)
    if readout_mode == "Z":
        return q
    if readout_mode == "Z+ZZ":
        d = q + max(q - 1, 0)
        if include_ring_pair and q > 2:
            d += 1
        return d
    raise ValueError("readout_mode must be 'Z' or 'Z+ZZ'")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="(7) Word2Ket-based encoding (no fallback)")
    p.add_argument("--outdir", type=str, default="runs", help="Output directory (results + artifacts).")
    p.add_argument("--cache_dir", type=str, default="cache", help="Kept for parity; Word2Ket pathing is handled by the library.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, default=0, help="0 -> analytic (None), else number of shots.")
    p.add_argument("--device", type=str, default="default.qubit")

    p.add_argument("--w2k_encoding", type=str, required=True, choices=["angle", "amplitude"])
    p.add_argument("--q", "--n_qubits", dest="q", type=int, default=8, help="Number of qubits (q).")

    p.add_argument("--entangler", type=str, default="none", choices=["none", "cnot", "cz"])
    p.add_argument("--topology", type=str, default="chain", choices=["chain", "ring"])
    p.add_argument("--readout", type=str, default="Z", choices=["Z", "Z+ZZ"])
    p.add_argument("--include_ring_pair", action="store_true")

    p.add_argument(
        "--w2k_cfg_json",
        type=str,
        default="{}",
        help="JSON dict of Word2Ket hyperparameters (passed to the Word2Ket constructor).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results_csv = os.path.join(args.outdir, "results_checklist_all_methods.csv")
    results_json = os.path.join(args.outdir, "results_checklist_all_methods.json")

    shots = None if int(args.shots) == 0 else int(args.shots)
    q = int(args.q)

    try:
        w2k_cfg = json.loads(args.w2k_cfg_json)
        if w2k_cfg is None:
            w2k_cfg = {}
        if not isinstance(w2k_cfg, dict):
            raise ValueError
    except Exception as e:
        raise ValueError(f"--w2k_cfg_json must be a JSON object (dict). Got: {args.w2k_cfg_json!r}") from e

    data = common.load_tweeteval_sentiment()
    texts_train = [str(x) for x in data["X_text_train"].tolist()]
    texts_val = [str(x) for x in data["X_text_val"].tolist()]
    texts_test = [str(x) for x in data["X_text_test"].tolist()]

    # -------------------------
    # Preprocessing: Word2Ket embedding (mandatory)
    # -------------------------
    t0 = time.perf_counter()
    Xtr_w2k = common.word2ket_embed_texts(
        texts_train,
        mode=str(args.w2k_encoding),
        q=q,
        w2k_cfg=w2k_cfg,
        seed=int(args.seed),
    )

    Xva_w2k = common.word2ket_embed_texts(
        texts_val,
        mode=str(args.w2k_encoding),
        q=q,
        w2k_cfg=w2k_cfg,
        seed=int(args.seed),
    )

    Xte_w2k = common.word2ket_embed_texts(
        texts_test,
        mode=str(args.w2k_encoding),
        q=q,
        w2k_cfg=w2k_cfg,
        seed=int(args.seed),
    )

    preproc_time_s = time.perf_counter() - t0

    # -------------------------
    # QNode
    # -------------------------
    ent = str(args.entangler)
    topo = str(args.topology)
    readout = str(args.readout)
    include_ring_pair = bool(args.include_ring_pair)

    if args.w2k_encoding == "angle":
        qnode = common.make_qnode_word2ket_angle(
            q,
            axis="ry",
            entangler=ent,
            topology=topo,
            readout_mode=readout,
            include_ring_pair=include_ring_pair,
            device_name=str(args.device),
            shots=shots,
        )
        pattern_for_proxy = "word2ket_angle_none" if ent == "none" else "word2ket_angle_linear"
        entanglement_label = f"word2ket-angle|{ent}|{topo}" if ent != "none" else "word2ket-angle|none"
    else:
        qnode = common.make_qnode_word2ket_amplitude(
            q,
            entangler=ent,
            topology=topo,
            readout_mode=readout,
            include_ring_pair=include_ring_pair,
            device_name=str(args.device),
            shots=shots,
        )
        pattern_for_proxy = "word2ket_amplitude_none" if ent == "none" else "word2ket_amplitude_linear"
        entanglement_label = f"word2ket-amplitude|{ent}|{topo}" if ent != "none" else "word2ket-amplitude|none"

    feature_dim = _feature_dim_from_readout(q, readout, include_ring_pair)

    technique_name = f"(7) word2ket | enc={args.w2k_encoding} | readout={readout} | ent={ent} | topo={topo} | q={q}"

    row = common.run_and_save(
        technique_name=technique_name,
        qnode=qnode,
        X_train_in=Xtr_w2k,
        X_val_in=Xva_w2k,
        X_test_in=Xte_w2k,
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=q,
        L=1,
        entanglement_pattern_label=entanglement_label,
        feature_dim=feature_dim,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=preproc_time_s,
        pattern_for_proxy=pattern_for_proxy,
        topology=topo,
        rot_mode="single",
        block_size_for_proxy=2,
        time_ref_X=Xtr_w2k,
        extra_fields={
            "method": "word2ket",
            "w2k_encoding": str(args.w2k_encoding),
            "k": int(2 ** q) if args.w2k_encoding == "amplitude" else None,
            "readout_mode": readout,
            "entangler": ent,
            "topology": topo,
            "include_ring_pair": include_ring_pair,
        },
    )

    print("Saved:", row)


if __name__ == "__main__":
    main()
