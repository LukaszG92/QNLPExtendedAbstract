from __future__ import annotations

import json
import os
import time

from common import (
    build_common_argparser,
    cfg_from_args,
    load_tweeteval_sentiment,
    word2ket_embed_texts,
    make_qnode_word2ket_angle,
    make_qnode_word2ket_amplitude,
    run_and_save,
    extra_fields_from_args,
)


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


def _as_text_list(x) -> list[str]:
    # Robust against pandas Series / numpy arrays / python lists.
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [str(t) for t in list(x)]


def main() -> None:
    p = build_common_argparser("(4) Word2Ket (mandatory library; no fallback)")

    # Word2Ket-specific knobs (kept minimal and explicit)
    p.add_argument(
        "--w2k_encoding",
        type=str,
        default="angle",
        choices=["angle", "amplitude"],
        help="Word2Ket output mode. 'angle' -> length q; 'amplitude' -> length 2^q.",
    )
    p.add_argument(
        "--w2k_cfg_json",
        type=str,
        default="{}",
        help="JSON object with Word2Ket hyperparameters (passed to the Word2Ket constructor).",
    )
    p.add_argument(
        "--include_ring_pair",
        action="store_true",
        help="If readout=Z+ZZ and topology=ring, also include the (q-1,0) ZZ pair in the readout.",
    )

    args = p.parse_args()
    cfg = cfg_from_args(args)

    if os.environ.get("QNLP_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on"):
        print("[DEBUG] tech04_word2ket: args=", vars(args))
        print("[DEBUG] tech04_word2ket: cfg=", cfg)

    # q for Word2Ket; allow a dedicated cfg field if you added it, otherwise fall back to n_qubits.
    q = int(getattr(cfg, "q_w2k", getattr(cfg, "n_qubits", 0)))
    if q <= 0:
        raise ValueError("Invalid q for Word2Ket. Expected cfg.q_w2k (or cfg.n_qubits) to be a positive integer.")

    # Sensible defaults for Word2Ket hyperparameters.
    # NOTE: num_embeddings is dataset/vocab-dependent; it must be injected by common.py.
    if args.w2k_cfg_json is None or str(args.w2k_cfg_json).strip() in ("", "{}"):
        args.w2k_cfg_json = json.dumps(
            {
                "embedding_dim": int(q),
                "order": 4,
                "rank": 1,
                "use_xs": True,
            }
        )

    # Parse Word2Ket config JSON (mandatory library; any import/constructor issues must surface)
    try:
        w2k_cfg = json.loads(args.w2k_cfg_json)
        if w2k_cfg is None:
            w2k_cfg = {}
        if not isinstance(w2k_cfg, dict):
            raise ValueError
    except Exception as e:
        raise ValueError(f"--w2k_cfg_json must be a JSON object (dict). Got: {args.w2k_cfg_json!r}") from e

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    results_csv = os.path.join(outdir, "results_checklist_all_methods.csv")
    results_json = os.path.join(outdir, "results_checklist_all_methods.json")

    data = load_tweeteval_sentiment()
    texts_train = _as_text_list(data["X_text_train"])
    texts_val = _as_text_list(data["X_text_val"])
    texts_test = _as_text_list(data["X_text_test"])

    # -------------------------
    # Preprocessing: Word2Ket embedding (mandatory; no fallback)
    # -------------------------
    t0 = time.perf_counter()
    Xtr = word2ket_embed_texts(
        texts_train,
        mode=str(args.w2k_encoding),
        q=q,
        w2k_cfg=w2k_cfg,
        # IMPORTANT: do NOT pass a seed into the Word2Ket ctor unless common.py explicitly supports it.
        # Reproducibility should be controlled via global seeds (torch/numpy/random) inside common.py.
        seed=None,
    )
    Xva = word2ket_embed_texts(
        texts_val,
        mode=str(args.w2k_encoding),
        q=q,
        w2k_cfg=w2k_cfg,
        seed=None,
    )
    Xte = word2ket_embed_texts(
        texts_test,
        mode=str(args.w2k_encoding),
        q=q,
        w2k_cfg=w2k_cfg,
        seed=None,
    )
    preproc_time_s = time.perf_counter() - t0

    if os.environ.get("QNLP_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on"):
        print(
            "[DEBUG] prep shapes:",
            "X_train_w2k",
            getattr(Xtr, "shape", None),
            "X_val_w2k",
            getattr(Xva, "shape", None),
            "X_test_w2k",
            getattr(Xte, "shape", None),
        )

    # -------------------------
    # QNode
    # -------------------------
    ent = str(getattr(cfg, "entangler", "none"))
    topo = str(getattr(cfg, "topology", "chain"))

    # Common config may expose either cfg.readout or cfg.readout_mode; accept both.
    readout = getattr(cfg, "readout_mode", getattr(cfg, "readout", "Z"))
    readout = str(readout)

    include_ring_pair = bool(getattr(args, "include_ring_pair", False))

    if str(args.w2k_encoding) == "angle":
        # axis: reuse cfg.axis if present, otherwise default to 'ry' (tech07 used 'ry')
        axis = str(getattr(cfg, "axis", "ry"))
        qnode = make_qnode_word2ket_angle(
            q,
            L=int(getattr(cfg, 'L', 1)),
            axis=axis,
            entangler=ent,
            topology=topo,
            readout_mode=readout,
            include_ring_pair=include_ring_pair,
            device_name=cfg.device_name,
            shots=cfg.shots,
        )
        pattern_for_proxy = "word2ket_angle_none" if ent == "none" else "word2ket_angle_linear"
        entanglement_label = f"word2ket-angle|{ent}|{topo}" if ent != "none" else "word2ket-angle|none"
    else:
        qnode = make_qnode_word2ket_amplitude(
            q,
            L=int(getattr(cfg, 'L', 1)),
            entangler=ent,
            topology=topo,
            readout_mode=readout,
            include_ring_pair=include_ring_pair,
            device_name=cfg.device_name,
            shots=cfg.shots,
        )
        pattern_for_proxy = "word2ket_amplitude_none" if ent == "none" else "word2ket_amplitude_linear"
        entanglement_label = f"word2ket-amplitude|{ent}|{topo}" if ent != "none" else "word2ket-amplitude|none"

    feature_dim = _feature_dim_from_readout(q, readout, include_ring_pair)

    extra = extra_fields_from_args(args)
    # Ensure a few canonical keys are always present for downstream aggregation
    extra.update(
        {
            "method": "word2ket",
            "w2k_encoding": str(args.w2k_encoding),
            "w2k_cfg": w2k_cfg,
            "readout_mode": readout,
            "entangler": ent,
            "topology": topo,
            "include_ring_pair": include_ring_pair,
            "k": int(2**q) if str(args.w2k_encoding) == "amplitude" else None,
        }
    )
    clf_kind = str(extra.get("clf_kind", "linearsvc")).lower()

    row = run_and_save(
        technique_name=(
            f"(4) word2ket | enc={args.w2k_encoding} | | L={cfg.L} | "
            f"ent={ent} | topo={topo} | q={q} | clf={clf_kind}"
        ),
        qnode=qnode,
        X_train_in=Xtr,
        X_val_in=Xva,
        X_test_in=Xte,
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        q=q,
        L=int(getattr(cfg, 'L', 1)),
        entanglement_pattern_label=entanglement_label,
        feature_dim=feature_dim,
        results_csv=results_csv,
        results_json=results_json,
        preproc_time_s=float(preproc_time_s),
        pattern_for_proxy=pattern_for_proxy,
        topology=topo,
        rot_mode=str(getattr(cfg, "rot_mode", "single")),
        block_size_for_proxy=2,
        time_ref_X=Xtr,
        extra_fields=extra,
    )

    print("Saved:", row)


if __name__ == "__main__":
    main()
