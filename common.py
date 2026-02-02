from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Tuple, Dict, Any

import numpy as np
import pennylane as qml
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

# =========================
# Mandatory dependency: word2ket
# =========================
try:
    import word2ket  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing required dependency 'word2ket'.\n"
        "Install it in your active environment (no fallback is provided).\n"
        "Example:\n"
        "  pip install word2ket\n"
        "If installation fails due to torch/gpytorch constraints, follow the word2ket project's instructions.\n"
        f"Original import error: {e!r}"
    ) from e


# =========================
# Config
# =========================

Entangler = Literal["cnot", "cz"]
Topology = Literal["chain", "ring"]
RotAxis = Literal["rx", "ry", "rz"]
RotMode = Literal["single", "xyz"]


@dataclass(frozen=True)
class BenchmarkConfig:
    # General
    random_state: int = 42
    device_name: str = "default.qubit"
    shots: Optional[int] = None

    # Angle encoding
    n_qubits: int = 8
    entangler: Entangler = "cnot"
    topology: Topology = "chain"
    rot_mode: RotMode = "single"
    axis: RotAxis = "ry"

    # Text -> TF-IDF -> SVD(q)
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

    # Amplitude encoding
    q_amp: int = 6  # k=2^q_amp
    # Lightweight compression
    k_base: int = 0  # if 0: max(4*n_qubits, 32)

    # MorphTE proxy
    char_ngram_range: Tuple[int, int] = (3, 5)


DATASET_NAME = "tweet_eval_sentiment"


# =========================
# I/O helpers
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_result_row(row: dict, csv_path: str, json_path: str) -> pd.DataFrame:
    """Append/replace a technique row on disk (idempotent by technique name)."""
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
        if "technique" in df.columns:
            df = df.drop_duplicates(subset=["technique"], keep="last")
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    return df


# =========================
# Dataset
# =========================

def load_tweeteval_sentiment() -> Dict[str, np.ndarray]:
    """Loads cardiffnlp/tweet_eval sentiment splits."""
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    return {
        "X_text_train": np.array(train_ds["text"], dtype=object),
        "y_train": np.array(train_ds["label"], dtype=int),
        "X_text_val": np.array(val_ds["text"], dtype=object),
        "y_val": np.array(val_ds["label"], dtype=int),
        "X_text_test": np.array(test_ds["text"], dtype=object),
        "y_test": np.array(test_ds["label"], dtype=int),
    }


# =========================
# Preprocessing
# =========================

def scale_angles(X: np.ndarray) -> np.ndarray:
    return (np.pi * np.tanh(X)).astype(np.float32, copy=False)


def tfidf_svd_angles(
    X_text_train: np.ndarray,
    X_text_val: np.ndarray,
    X_text_test: np.ndarray,
    *,
    n_components: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    random_state: int,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """TF-IDF(word) -> SVD(n_components) -> angles in [-pi, pi]."""
    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_path = os.path.join(
            cache_dir,
            f"tfidf_svd_angles_q{n_components}_ng{ngram_range[0]}{ngram_range[1]}_min{min_df}_max{max_df}_seed{random_state}.npz",
        )
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            return {
                "X_train_ang": data["X_train_ang"],
                "X_val_ang": data["X_val_ang"],
                "X_test_ang": data["X_test_ang"],
                "preproc_time_s": float(data["preproc_time_s"]),
            }

    t0 = time.perf_counter()
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    Xtr_tfidf = tfidf.fit_transform(X_text_train)
    Xva_tfidf = tfidf.transform(X_text_val)
    Xte_tfidf = tfidf.transform(X_text_test)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_train = svd.fit_transform(Xtr_tfidf).astype(np.float32, copy=False)
    X_val = svd.transform(Xva_tfidf).astype(np.float32, copy=False)
    X_test = svd.transform(Xte_tfidf).astype(np.float32, copy=False)

    X_train_ang = scale_angles(X_train)
    X_val_ang = scale_angles(X_val)
    X_test_ang = scale_angles(X_test)

    preproc_time_s = time.perf_counter() - t0

    if cache_path:
        np.savez_compressed(
            cache_path,
            X_train_ang=X_train_ang,
            X_val_ang=X_val_ang,
            X_test_ang=X_test_ang,
            preproc_time_s=np.array(preproc_time_s, dtype=np.float64),
        )

    return {
        "X_train_ang": X_train_ang,
        "X_val_ang": X_val_ang,
        "X_test_ang": X_test_ang,
        "preproc_time_s": float(preproc_time_s),
    }


def fix_zero_rows_unitvec(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure each row is a valid non-zero state vector for amplitude embedding."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    bad = norms < eps
    if np.any(bad):
        X = X.copy()
        X[bad, :] = 0.0
        X[bad, 0] = 1.0
    return X


def hashing_amp_vectors(
    X_text: np.ndarray,
    *,
    k_amp: int,
    random_state: int,
) -> np.ndarray:
    """
    text -> HashingVectorizer(k_amp) -> TF-IDF transform -> dense -> L2 normalize -> safe unit vectors.
    k_amp should be 2^q_amp.
    """
    hv = HashingVectorizer(
        n_features=k_amp,
        alternate_sign=False,
        norm=None,
        analyzer="word",
        ngram_range=(1, 2),
    )
    X_counts = hv.transform(X_text)
    tfidf_tr = TfidfTransformer()
    X_tfidf = tfidf_tr.fit_transform(X_counts)

    X_dense = X_tfidf.toarray().astype(np.float32, copy=False)
    X_dense = normalize(X_dense, norm="l2", axis=1)
    X_dense = fix_zero_rows_unitvec(X_dense)
    return X_dense.astype(np.float32, copy=False)


def light_compress_from_word_tfidf(
    X_text_train: np.ndarray,
    X_text_val: np.ndarray,
    X_text_test: np.ndarray,
    *,
    n_qubits: int,
    k_amp: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    random_state: int,
    k_base: int,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    (6) Lightweight compression:
      word TF-IDF -> SVD(k_base) -> random projection to q (angles) and to k_amp (amplitude) + normalization.
    """
    if k_base <= 0:
        k_base = int(max(4 * n_qubits, 32))

    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_path = os.path.join(
            cache_dir,
            f"lightcompress_kbase{k_base}_q{n_qubits}_kamp{k_amp}_ng{ngram_range[0]}{ngram_range[1]}_min{min_df}_max{max_df}_seed{random_state}.npz",
        )
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            return {
                "X_train_lc_ang": data["X_train_lc_ang"],
                "X_val_lc_ang": data["X_val_lc_ang"],
                "X_test_lc_ang": data["X_test_lc_ang"],
                "X_train_amp_lc": data["X_train_amp_lc"],
                "X_val_amp_lc": data["X_val_amp_lc"],
                "X_test_amp_lc": data["X_test_amp_lc"],
                "preproc_time_s": float(data["preproc_time_s"]),
                "k_base": int(data["k_base"]),
            }

    t0 = time.perf_counter()

    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    Xtr_tfidf = tfidf.fit_transform(X_text_train)
    Xva_tfidf = tfidf.transform(X_text_val)
    Xte_tfidf = tfidf.transform(X_text_test)

    svd_base = TruncatedSVD(n_components=k_base, random_state=random_state)
    X_train_base = svd_base.fit_transform(Xtr_tfidf).astype(np.float32, copy=False)
    X_val_base = svd_base.transform(Xva_tfidf).astype(np.float32, copy=False)
    X_test_base = svd_base.transform(Xte_tfidf).astype(np.float32, copy=False)

    def project_dense(X: np.ndarray, out_dim: int, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(X.shape[1], out_dim)).astype(np.float32)
        W /= np.sqrt(X.shape[1])
        return (X @ W).astype(np.float32, copy=False)

    # angle
    X_train_lc = project_dense(X_train_base, n_qubits, seed=random_state + 11)
    X_val_lc = project_dense(X_val_base, n_qubits, seed=random_state + 11)
    X_test_lc = project_dense(X_test_base, n_qubits, seed=random_state + 11)
    X_train_lc_ang = scale_angles(X_train_lc)
    X_val_lc_ang = scale_angles(X_val_lc)
    X_test_lc_ang = scale_angles(X_test_lc)

    # amplitude
    X_train_amp_lc = project_dense(X_train_base, k_amp, seed=random_state + 23)
    X_val_amp_lc = project_dense(X_val_base, k_amp, seed=random_state + 23)
    X_test_amp_lc = project_dense(X_test_base, k_amp, seed=random_state + 23)

    X_train_amp_lc = normalize(X_train_amp_lc, norm="l2", axis=1)
    X_val_amp_lc = normalize(X_val_amp_lc, norm="l2", axis=1)
    X_test_amp_lc = normalize(X_test_amp_lc, norm="l2", axis=1)

    X_train_amp_lc = fix_zero_rows_unitvec(X_train_amp_lc)
    X_val_amp_lc = fix_zero_rows_unitvec(X_val_amp_lc)
    X_test_amp_lc = fix_zero_rows_unitvec(X_test_amp_lc)

    preproc_time_s = time.perf_counter() - t0

    if cache_path:
        np.savez_compressed(
            cache_path,
            X_train_lc_ang=X_train_lc_ang,
            X_val_lc_ang=X_val_lc_ang,
            X_test_lc_ang=X_test_lc_ang,
            X_train_amp_lc=X_train_amp_lc,
            X_val_amp_lc=X_val_amp_lc,
            X_test_amp_lc=X_test_amp_lc,
            preproc_time_s=np.array(preproc_time_s, dtype=np.float64),
            k_base=np.array(k_base, dtype=np.int64),
        )

    return {
        "X_train_lc_ang": X_train_lc_ang,
        "X_val_lc_ang": X_val_lc_ang,
        "X_test_lc_ang": X_test_lc_ang,
        "X_train_amp_lc": X_train_amp_lc,
        "X_val_amp_lc": X_val_amp_lc,
        "X_test_amp_lc": X_test_amp_lc,
        "preproc_time_s": float(preproc_time_s),
        "k_base": int(k_base),
    }


def morphte_char_angles(
    X_text_train: np.ndarray,
    X_text_val: np.ndarray,
    X_text_test: np.ndarray,
    *,
    n_qubits: int,
    char_ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    random_state: int,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    (8) MorphTE-inspired proxy:
      char_wb TF-IDF (subword) -> SVD(q) -> angles
    """
    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_path = os.path.join(
            cache_dir,
            f"morphte_char_angles_q{n_qubits}_ng{char_ngram_range[0]}{char_ngram_range[1]}_min{min_df}_max{max_df}_seed{random_state}.npz",
        )
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            return {
                "X_train_char_ang": data["X_train_char_ang"],
                "X_val_char_ang": data["X_val_char_ang"],
                "X_test_char_ang": data["X_test_char_ang"],
                "preproc_time_s": float(data["preproc_time_s"]),
            }

    t0 = time.perf_counter()
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    Xtr_char = tfidf_char.fit_transform(X_text_train)
    Xva_char = tfidf_char.transform(X_text_val)
    Xte_char = tfidf_char.transform(X_text_test)

    svd_char = TruncatedSVD(n_components=n_qubits, random_state=random_state)
    X_train_char = svd_char.fit_transform(Xtr_char).astype(np.float32, copy=False)
    X_val_char = svd_char.transform(Xva_char).astype(np.float32, copy=False)
    X_test_char = svd_char.transform(Xte_char).astype(np.float32, copy=False)

    X_train_char_ang = scale_angles(X_train_char)
    X_val_char_ang = scale_angles(X_val_char)
    X_test_char_ang = scale_angles(X_test_char)

    preproc_time_s = time.perf_counter() - t0

    if cache_path:
        np.savez_compressed(
            cache_path,
            X_train_char_ang=X_train_char_ang,
            X_val_char_ang=X_val_char_ang,
            X_test_char_ang=X_test_char_ang,
            preproc_time_s=np.array(preproc_time_s, dtype=np.float64),
        )

    return {
        "X_train_char_ang": X_train_char_ang,
        "X_val_char_ang": X_val_char_ang,
        "X_test_char_ang": X_test_char_ang,
        "preproc_time_s": float(preproc_time_s),
    }


# =========================
# Circuits
# =========================

def _apply_rotation(axis: RotAxis, angle: float, wire: int) -> None:
    if axis == "rx":
        qml.RX(angle, wires=wire)
    elif axis == "ry":
        qml.RY(angle, wires=wire)
    elif axis == "rz":
        qml.RZ(angle, wires=wire)
    else:
        raise ValueError(f"Unknown axis: {axis}")


def _apply_entangler(kind: Entangler, control: int, target: int) -> None:
    if kind == "cnot":
        qml.CNOT(wires=[control, target])
    elif kind == "cz":
        qml.CZ(wires=[control, target])
    else:
        raise ValueError(f"Unknown entangler: {kind}")


def _pad_trunc(x: np.ndarray, n_qubits: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < n_qubits:
        return np.pad(x, (0, n_qubits - x.size))
    return x[:n_qubits]


def _angle_encode(x: np.ndarray, *, n_qubits: int, rot_mode: RotMode, axis: RotAxis) -> None:
    x = np.asarray(x, dtype=float).ravel()
    if rot_mode == "single":
        x_used = _pad_trunc(x, n_qubits)
        for w in range(n_qubits):
            _apply_rotation(axis, float(x_used[w]), w)
    elif rot_mode == "xyz":
        needed = 3 * n_qubits
        if x.size < needed:
            x3 = np.pad(x, (0, needed - x.size))
        else:
            x3 = x[:needed]
        x3 = x3.reshape(n_qubits, 3)
        for w in range(n_qubits):
            qml.RX(float(x3[w, 0]), wires=w)
            qml.RY(float(x3[w, 1]), wires=w)
            qml.RZ(float(x3[w, 2]), wires=w)
    else:
        raise ValueError(f"Unknown rot_mode: {rot_mode}")


def _entangle_linear(*, n_qubits: int, entangler: Entangler, topology: Topology) -> None:
    for i in range(n_qubits - 1):
        _apply_entangler(entangler, i, i + 1)
    if topology == "ring" and n_qubits > 2:
        _apply_entangler(entangler, n_qubits - 1, 0)
    elif topology == "chain":
        pass
    elif topology != "ring":
        raise ValueError("topology must be 'chain' or 'ring'")


def _entangle_all_to_all(*, n_qubits: int, entangler: Entangler) -> None:
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            _apply_entangler(entangler, i, j)


def _entangle_block(*, n_qubits: int, entangler: Entangler, block_size: int) -> None:
    b = int(block_size)
    if b < 2:
        _entangle_linear(n_qubits=n_qubits, entangler=entangler, topology="chain")
        return

    for start in range(0, n_qubits, b):
        end = min(start + b, n_qubits)
        # within-block all-to-all
        for i in range(start, end):
            for j in range(i + 1, end):
                _apply_entangler(entangler, i, j)
        # single link between blocks
        if end < n_qubits:
            _apply_entangler(entangler, end - 1, end)


# =========================
# Word2Ket bridge (document embedding)
# =========================

def _w2k_find_embedder() -> Any:
    """Best-effort discovery of a Word2Ket embedder object.

    This intentionally does *not* implement any fallback embedding logic.
    If the Word2Ket API cannot be discovered/used, we raise a clear error.
    """
    # Common candidates at top-level
    for name in ("Word2Ket", "EmbeddingKet", "EmbeddingKets", "Word2Kets"):
        if hasattr(word2ket, name):
            return getattr(word2ket, name)

    # Try to discover in submodules (robust to API moves)
    try:
        import pkgutil
        import importlib

        if hasattr(word2ket, "__path__"):
            for m in pkgutil.walk_packages(word2ket.__path__, word2ket.__name__ + "."):
                try:
                    mod = importlib.import_module(m.name)
                except Exception:
                    continue
                for name in ("Word2Ket", "EmbeddingKet", "EmbeddingKets", "Word2Kets"):
                    if hasattr(mod, name):
                        return getattr(mod, name)
    except Exception:
        pass

    raise RuntimeError(
        "Could not locate a Word2Ket class/function inside the installed 'word2ket' package. "
        "Expected something like word2ket.Word2Ket or word2ket.EmbeddingKet. "
        "Please inspect your installed word2ket version and update common.word2ket_embed_texts accordingly."
    )


def _w2k_instantiate(embedder_cls_or_fn: Any, *, w2k_cfg: Optional[Dict[str, Any]], seed: int) -> Any:
    w2k_cfg = {} if w2k_cfg is None else dict(w2k_cfg)
    # Seed handling if supported
    if "seed" not in w2k_cfg and "random_state" not in w2k_cfg:
        w2k_cfg["seed"] = seed
    try:
        # Class constructor
        return embedder_cls_or_fn(**w2k_cfg)
    except TypeError:
        # Might be a function factory or expects positional args
        try:
            return embedder_cls_or_fn(w2k_cfg)
        except Exception as e:
            raise RuntimeError(
                "Failed to instantiate Word2Ket embedder with w2k_cfg="
                f"{w2k_cfg}. Underlying error: {e!r}"
            ) from e


def _w2k_embed(embedder: Any, texts: list[str]) -> Any:
    """Call the embedder using common method names."""
    # Most likely methods
    for meth in ("encode", "embed", "transform", "vectorize", "__call__"):
        if meth == "__call__":
            fn = embedder
        else:
            if not hasattr(embedder, meth):
                continue
            fn = getattr(embedder, meth)
        try:
            out = fn(texts)
            if out is not None:
                return out
        except Exception:
            continue

    raise RuntimeError(
        "Installed Word2Ket object does not expose a usable embedding method for a list[str]. "
        "Tried: encode/embed/transform/vectorize/callable. "
        "Please adapt _w2k_embed() to your word2ket API."
    )


def _to_1d_real(x: Any) -> np.ndarray:
    """Convert arbitrary Word2Ket output to a 1D real numpy vector (deterministic)."""
    # Torch tensor support (without importing torch explicitly)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=float)
    return arr.ravel()


def word2ket_embed_texts(
    texts: list[str],
    *,
    mode: str,              # "angle" | "amplitude"
    q: int,
    w2k_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return Word2Ket embeddings as numpy arrays, adapted for circuit encodings.

    Returns:
      - mode="angle": shape (n, q) values already scaled to angles in [-pi, pi]
      - mode="amplitude": shape (n, 2**q) L2-normalized vectors (normalize=False in circuit)
    """
    if mode not in ("angle", "amplitude"):
        raise ValueError("mode must be 'angle' or 'amplitude'")
    if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
        raise TypeError("texts must be a list[str]")
    q = int(q)
    if q <= 0:
        raise ValueError("q must be a positive integer")

    embedder_cls_or_fn = _w2k_find_embedder()
    embedder = _w2k_instantiate(embedder_cls_or_fn, w2k_cfg=w2k_cfg, seed=seed)

    raw = _w2k_embed(embedder, texts)

    # raw may be (n, d) or list-like; normalize to list of vectors
    if isinstance(raw, (list, tuple)):
        rows = list(raw)
    else:
        arr = np.asarray(raw, dtype=object)
        if arr.ndim == 1 and len(texts) == 1:
            rows = [raw]
        elif arr.ndim >= 2:
            rows = [arr[i] for i in range(arr.shape[0])]
        else:
            raise RuntimeError(
                "Word2Ket returned an output with unsupported shape/type: "
                f"type={type(raw)}, ndim={getattr(arr, 'ndim', None)}"
            )

    if len(rows) != len(texts):
        # Some APIs return a single matrix; try to coerce
        try:
            mat = np.asarray(raw, dtype=float)
            if mat.ndim == 2 and mat.shape[0] == len(texts):
                rows = [mat[i] for i in range(mat.shape[0])]
            else:
                raise ValueError
        except Exception as e:
            raise RuntimeError(
                "Word2Ket embedding batch size mismatch: "
                f"expected n={len(texts)}, got {len(rows)}. Underlying: {e!r}"
            ) from e

    n = len(texts)
    if mode == "angle":
        out = np.zeros((n, q), dtype=np.float32)
        for i, r in enumerate(rows):
            v = _to_1d_real(r)
            # deterministic adaptation: slice or zero-pad
            if v.size >= q:
                u = v[:q]
            else:
                u = np.zeros(q, dtype=float)
                u[: v.size] = v
            # scale to angles in [-pi, pi] deterministically
            # Use tanh squashing to be robust to scale changes across word2ket versions
            ang = np.tanh(u) * math.pi
            out[i] = ang.astype(np.float32, copy=False)
        return out

    # mode == "amplitude"
    k = 2 ** q
    out = np.zeros((n, k), dtype=np.float32)
    for i, r in enumerate(rows):
        v = _to_1d_real(r)
        if v.size >= k:
            u = v[:k]
        else:
            u = np.zeros(k, dtype=float)
            u[: v.size] = v

        # L2-normalize; deterministic fix for null vectors
        norm = float(np.linalg.norm(u))
        if not np.isfinite(norm):
            raise ValueError(f"Non-finite norm in Word2Ket amplitude vector at i={i}")
        if norm == 0.0:
            u[0] = 1.0
            norm = 1.0
        u = u / norm
        out[i] = u.astype(np.float32, copy=False)
    return out


# =========================
# QNode builders: Word2Ket encodings
# =========================

def _readout_observables(
    n_qubits: int,
    *,
    readout_mode: Literal["Z", "Z+ZZ"],
    include_ring_pair: bool,
):
    if readout_mode == "Z":
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
    if readout_mode == "Z+ZZ":
        obs = [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        for i in range(n_qubits - 1):
            obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
        if include_ring_pair and n_qubits > 2:
            obs.append(qml.expval(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0)))
        return obs
    raise ValueError("readout_mode must be 'Z' or 'Z+ZZ'")


def make_qnode_word2ket_angle(
    n_qubits: int,
    *,
    axis: str = "ry",
    entangler: str = "none",      # "none" | "cnot" | "cz"
    topology: str = "chain",      # "chain" | "ring"
    readout_mode: str = "Z",      # "Z" | "Z+ZZ"
    include_ring_pair: bool = False,
    device_name: str = "default.qubit",
    shots: Optional[int] = None,
):
    """Angle rotations on all qubits (1 layer), optional linear entanglement, configurable readout."""
    n_qubits = int(n_qubits)
    dev = qml.device(device_name, wires=n_qubits, shots=shots)
    axis_l = str(axis).lower()
    if axis_l not in ("rx", "ry", "rz"):
        raise ValueError("axis must be one of: rx, ry, rz")
    ent = str(entangler).lower()
    topo = str(topology).lower()
    read = str(readout_mode)

    @qml.qnode(dev)
    def qnode(x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size != n_qubits:
            raise ValueError(f"Expected angle input of size {n_qubits}, got {x.size}")
        for w in range(n_qubits):
            if axis_l == "rx":
                qml.RX(x[w], wires=w)
            elif axis_l == "ry":
                qml.RY(x[w], wires=w)
            else:
                qml.RZ(x[w], wires=w)

        if ent != "none":
            if ent not in ("cnot", "cz"):
                raise ValueError("entangler must be one of: none, cnot, cz")
            if topo not in ("chain", "ring"):
                raise ValueError("topology must be one of: chain, ring")
            _entangle_linear(n_qubits=n_qubits, entangler=ent, topology=topo)  # type: ignore[arg-type]

        return _readout_observables(n_qubits, readout_mode=read, include_ring_pair=bool(include_ring_pair))

    return qnode


def make_qnode_word2ket_amplitude(
    n_qubits: int,
    *,
    entangler: str = "none",
    topology: str = "chain",
    readout_mode: str = "Z",
    include_ring_pair: bool = False,
    device_name: str = "default.qubit",
    shots: Optional[int] = None,
):
    """AmplitudeEmbedding + optional linear entanglement, configurable readout."""
    n_qubits = int(n_qubits)
    dev = qml.device(device_name, wires=n_qubits, shots=shots)
    ent = str(entangler).lower()
    topo = str(topology).lower()
    read = str(readout_mode)

    @qml.qnode(dev)
    def qnode(x):
        x = np.asarray(x, dtype=float).ravel()
        expected = 2 ** n_qubits
        if x.size != expected:
            raise ValueError(f"Expected amplitude input of size {expected}, got {x.size}")
        qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=False)

        if ent != "none":
            if ent not in ("cnot", "cz"):
                raise ValueError("entangler must be one of: none, cnot, cz")
            if topo not in ("chain", "ring"):
                raise ValueError("topology must be one of: chain, ring")
            _entangle_linear(n_qubits=n_qubits, entangler=ent, topology=topo)  # type: ignore[arg-type]

        return _readout_observables(n_qubits, readout_mode=read, include_ring_pair=bool(include_ring_pair))

    return qnode


def make_qnode_angle(
    *,
    n_qubits: int,
    rot_mode: RotMode,
    axis: RotAxis,
    entangler: Entangler,
    topology: Topology,
    ent_pattern: Literal["linear", "all_to_all", "block", "none"],
    L: int = 1,
    block_size: int = 2,
    readout_mode: Literal["Z", "Z+ZZ"] = "Z",
    include_ring_pair: bool = False,
    device_name: str = "default.qubit",
    shots: Optional[int] = None,
):
    dev = qml.device(device_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def qnode(x):
        if ent_pattern == "none":
            _angle_encode(x, n_qubits=n_qubits, rot_mode=rot_mode, axis=axis)
        elif ent_pattern in ("linear", "all_to_all", "block"):
            # data re-uploading: repeat L blocks
            for _ in range(int(L)):
                _angle_encode(x, n_qubits=n_qubits, rot_mode=rot_mode, axis=axis)
                if ent_pattern == "linear":
                    _entangle_linear(n_qubits=n_qubits, entangler=entangler, topology=topology)
                elif ent_pattern == "all_to_all":
                    _entangle_all_to_all(n_qubits=n_qubits, entangler=entangler)
                elif ent_pattern == "block":
                    _entangle_block(n_qubits=n_qubits, entangler=entangler, block_size=block_size)
        else:
            raise ValueError("Unknown ent_pattern")

        if readout_mode == "Z":
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        if readout_mode == "Z+ZZ":
            obs = [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
            for i in range(n_qubits - 1):
                obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
            if include_ring_pair and n_qubits > 2:
                obs.append(qml.expval(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0)))
            return obs

        raise ValueError("readout_mode must be 'Z' or 'Z+ZZ'")

    return qnode

# =========================
# Trainable VQC builders (encoding + small variational head)
# =========================

def _variational_layer(
    weights: np.ndarray,
    *,
    n_qubits: int,
    entangler: Entangler,
    topology: Topology,
) -> None:
    """
    weights shape: (n_qubits, 3), applied as qml.Rot(phi, theta, omega) per qubit.
    """
    for w in range(n_qubits):
        qml.Rot(weights[w, 0], weights[w, 1], weights[w, 2], wires=w)
    _entangle_linear(n_qubits=n_qubits, entangler=entangler, topology=topology)


def make_trainable_qnode_angle(
    *,
    n_qubits: int,
    rot_mode: RotMode,
    axis: RotAxis,
    entangler: Entangler,
    topology: Topology,
    ent_pattern: Literal["linear", "all_to_all", "block", "none"],
    L: int = 1,
    block_size: int = 2,
    readout_mode: Literal["Z", "Z+ZZ"] = "Z",
    include_ring_pair: bool = False,
    var_layers: int = 1,                      # trainable depth
    var_entangler: Optional[Entangler] = None,
    device_name: str = "default.qubit",
    shots: Optional[int] = None,
):
    """
    Trainable QNode: (encoding + ent_pattern) + var_layers of Rot+linear entanglement.
    Signature: qnode(x, w) where w has shape (var_layers, n_qubits, 3).
    """
    dev = qml.device(device_name, wires=n_qubits, shots=shots)
    ve = entangler if var_entangler is None else var_entangler

    @qml.qnode(dev, interface="autograd")
    def qnode(x, w):
        # ---- fixed encoding block (same as make_qnode_angle) ----
        if ent_pattern == "none":
            _angle_encode(x, n_qubits=n_qubits, rot_mode=rot_mode, axis=axis)
        elif ent_pattern in ("linear", "all_to_all", "block"):
            for _ in range(int(L)):
                _angle_encode(x, n_qubits=n_qubits, rot_mode=rot_mode, axis=axis)
                if ent_pattern == "linear":
                    _entangle_linear(n_qubits=n_qubits, entangler=entangler, topology=topology)
                elif ent_pattern == "all_to_all":
                    _entangle_all_to_all(n_qubits=n_qubits, entangler=entangler)
                else:
                    _entangle_block(n_qubits=n_qubits, entangler=entangler, block_size=block_size)
        else:
            raise ValueError("Unknown ent_pattern")

        # ---- trainable variational head ----
        for l in range(int(var_layers)):
            _variational_layer(w[l], n_qubits=n_qubits, entangler=ve, topology=topology)

        # ---- readout ----
        if readout_mode == "Z":
            return [qml.expval(qml.PauliZ(wi)) for wi in range(n_qubits)]

        if readout_mode == "Z+ZZ":
            obs = [qml.expval(qml.PauliZ(wi)) for wi in range(n_qubits)]
            for i in range(n_qubits - 1):
                obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
            if include_ring_pair and n_qubits > 2:
                obs.append(qml.expval(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0)))
            return obs

        raise ValueError("readout_mode must be 'Z' or 'Z+ZZ'")

    return qnode

def make_qnode_amplitude(
    *,
    n_qubits: int,
    device_name: str = "default.qubit",
    shots: Optional[int] = None,
):
    dev = qml.device(device_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def qnode(x):
        x = np.asarray(x, dtype=float).ravel()
        # normalize=False because we already normalized in preprocessing and fixed zero rows.
        qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=False)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return qnode


# =========================
# Encoding + timing
# =========================

def encode_dataset_with_qnode(
    qnode,
    X: np.ndarray,
    *,
    out_dim: Optional[int] = None,
    catch_exceptions: bool = True,
) -> Dict[str, Any]:
    """
    Encodes samples with qnode.
    Returns dict: Xq, n_fail (samples that failed and were set to zeros).
    """
    X = np.asarray(X, dtype=float)
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("Empty X")

    # Determine output dim
    y0 = np.asarray(qnode(X[0]), dtype=float).ravel()
    d = int(out_dim) if out_dim is not None else int(y0.size)

    out = np.zeros((n, d), dtype=np.float32)
    out[0] = y0.astype(np.float32, copy=False)

    n_fail = 0
    for i in range(1, n):
        try:
            yi = np.asarray(qnode(X[i]), dtype=float).ravel()
            if yi.size != d:
                raise ValueError(f"QNode output dim mismatch at i={i}: got {yi.size}, expected {d}")
            out[i] = yi.astype(np.float32, copy=False)
        except Exception:
            if not catch_exceptions:
                raise
            n_fail += 1
            # keep zeros row

    # Defensive: remove NaN/Inf
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return {"Xq": out, "n_fail": int(n_fail)}


def time_encoding_per_sample(qnode, X: np.ndarray, n_warmup: int = 10, n_timed: int = 200) -> float:
    X = np.asarray(X, dtype=float)
    n = int(X.shape[0])
    if n == 0:
        return float("nan")
    n_warmup = min(n_warmup, n)
    n_timed = min(n_timed, n)

    for i in range(n_warmup):
        _ = qnode(X[i])

    t0 = time.perf_counter()
    for i in range(n_timed):
        _ = qnode(X[i])
    t1 = time.perf_counter()
    return (t1 - t0) / max(n_timed, 1)


# =========================
# Metrics + resources
# =========================

def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def get_backend_name(qnode) -> str:
    dev = getattr(qnode, "device", None)
    if dev is None:
        return "unknown"
    return getattr(dev, "short_name", None) or getattr(dev, "name", None) or dev.__class__.__name__


def get_shots(qnode):
    dev = getattr(qnode, "device", None)
    if dev is None:
        return None
    return getattr(dev, "shots", None)


def safe_specs(qnode, x_sample: np.ndarray) -> Dict[str, Any]:
    try:
        return qml.specs(qnode)(x_sample)
    except Exception as e:
        return {"error": str(e)}


_TWOQ_NAMES = {
    "CNOT", "CZ", "SWAP", "ISWAP", "CSWAP",
    "CRX", "CRY", "CRZ",
    "ControlledPhaseShift", "ControlledQubitUnitary",
    "IsingXX", "IsingYY", "IsingZZ",
}


def _twoq_from_gate_types(gate_types: Any) -> Optional[int]:
    if not isinstance(gate_types, dict):
        return None
    total = 0
    found = False
    for k, v in gate_types.items():
        if k in _TWOQ_NAMES:
            total += int(v)
            found = True
    return int(total) if found else None


def twoq_gates_proxy(n_qubits: int, pattern: str, *, topology: str = "chain", L: int = 1, block_size: int = 2) -> int:
    if pattern == "linear":
        if topology == "chain":
            return max(n_qubits - 1, 0)
        if topology == "ring":
            return n_qubits if n_qubits > 2 else max(n_qubits - 1, 0)
        return max(n_qubits - 1, 0)

    if pattern == "all_to_all":
        return int(n_qubits * (n_qubits - 1) // 2)

    if pattern == "reupload_linear":
        return int(L) * twoq_gates_proxy(n_qubits, "linear", topology=topology, L=1)

    if pattern == "word2ket_angle_none":
        return 0

    if pattern == "word2ket_angle_linear":
        return twoq_gates_proxy(n_qubits, "linear", topology=topology, L=1)

    if pattern == "word2ket_amplitude_none":
        return 0

    if pattern == "word2ket_amplitude_linear":
        return int(twoq_gates_proxy(n_qubits, "amplitude_stateprep") + twoq_gates_proxy(n_qubits, "linear", topology=topology, L=1))

    if pattern == "block":
        b = int(block_size)
        if b < 2:
            return twoq_gates_proxy(n_qubits, "linear", topology=topology, L=L)
        total = 0
        for start in range(0, n_qubits, b):
            end = min(start + b, n_qubits)
            m = end - start
            total += m * (m - 1) // 2
            if end < n_qubits:
                total += 1
        return int(total)

    if pattern == "amplitude_stateprep":
        # Very coarse proxy: O(2^q)
        return max(int(2 ** n_qubits - n_qubits - 1), 0)

    return 0


def depth_proxy(n_qubits: int, pattern: str, *, rot_mode: str = "single", L: int = 1, block_size: int = 2) -> int:
    enc_depth = 1 if rot_mode == "single" else 3

    if pattern == "linear":
        return int(enc_depth + 1)
    if pattern == "all_to_all":
        return int(enc_depth + max(n_qubits - 1, 1))
    if pattern == "reupload_linear":
        return int(L) * int(enc_depth + 1)
    if pattern == "block":
        b = int(block_size)
        n_blocks = int(math.ceil(n_qubits / max(b, 1)))
        return int(enc_depth + max(b - 1, 1) + max(n_blocks - 1, 0))
    if pattern == "word2ket_angle_none":
        return int(enc_depth)

    if pattern == "word2ket_angle_linear":
        return int(enc_depth + 1)

    if pattern == "word2ket_amplitude_none":
        return int(2 ** n_qubits)

    if pattern == "word2ket_amplitude_linear":
        return int((2 ** n_qubits) + 1)

    if pattern == "amplitude_stateprep":
        return int(2 ** n_qubits)
    return int(enc_depth)


def resource_summary(
    qnode,
    *,
    x_sample: np.ndarray,
    n_qubits: int,
    pattern_for_proxy: str,
    topology: str,
    rot_mode: str,
    L: int,
    block_size_for_proxy: int,
) -> Dict[str, Any]:
    specs = safe_specs(qnode, x_sample)
    resources = specs.get("resources", {}) if isinstance(specs, dict) else {}

    depth = None
    twoq = None
    num_gates = None
    num_wires = None
    gate_sizes = None
    gate_types = None

    if isinstance(resources, dict):
        depth = resources.get("depth", None)
        gate_sizes = resources.get("gate_sizes", None) or {}
        gate_types = resources.get("gate_types", None)
        num_gates = resources.get("num_gates", None)
        num_wires = resources.get("num_wires", None)
        if isinstance(gate_sizes, dict):
            twoq = gate_sizes.get(2, None)
    else:
        depth = getattr(resources, "depth", None)
        gate_sizes = getattr(resources, "gate_sizes", None) or {}
        gate_types = getattr(resources, "gate_types", None)
        num_gates = getattr(resources, "num_gates", None)
        num_wires = getattr(resources, "num_wires", None)
        if isinstance(gate_sizes, dict):
            twoq = gate_sizes.get(2, None)

    if twoq is None:
        twoq = _twoq_from_gate_types(gate_types)

    if depth is None:
        depth = depth_proxy(n_qubits, pattern_for_proxy, rot_mode=rot_mode, L=L, block_size=block_size_for_proxy)

    if twoq is None:
        twoq = twoq_gates_proxy(n_qubits, pattern_for_proxy, topology=topology, L=L, block_size=block_size_for_proxy)

    return {
        "depth": depth,
        "two_qubit_gates": twoq,
        "num_gates": num_gates,
        "num_wires": num_wires,
        "gate_sizes": gate_sizes,
        "gate_types": gate_types,
        "specs_error": specs.get("error") if isinstance(specs, dict) else None,
    }


def make_checklist_row(
    *,
    dataset_name: str,
    technique_name: str,
    metrics_test: Dict[str, float],
    metrics_val: Optional[Dict[str, float]],
    q: int,
    L: int,
    entanglement_pattern: str,
    feature_dim: int,
    backend: str,
    shots,
    specs_resource: Dict[str, Any],
    enc_time_s_per_sample: float,
    preproc_time_s: Optional[float] = None,
    n_encode_fail: int = 0,
) -> Dict[str, Any]:
    row = {
        "dataset": dataset_name,
        "technique": technique_name,
        "macro_f1": float(metrics_test["macro_f1"]),
        "accuracy": float(metrics_test["acc"]),
        "val_macro_f1": None if metrics_val is None else float(metrics_val["macro_f1"]),
        "val_accuracy": None if metrics_val is None else float(metrics_val["acc"]),
        "q": int(q),
        "L": int(L),
        "depth": specs_resource.get("depth", None),
        "two_qubit_gates": specs_resource.get("two_qubit_gates", None),
        "entanglement_pattern": entanglement_pattern,
        "feature_dim": int(feature_dim),
        "encoding_time_s_per_sample": float(enc_time_s_per_sample),
        "preproc_time_s": None if preproc_time_s is None else float(preproc_time_s),
        "backend": backend,
        "shots": None if shots is None else int(shots) if isinstance(shots, (int, np.integer)) else str(shots),
        "n_encode_fail": int(n_encode_fail),
    }
    return row


# =========================
# Runner
# =========================

def run_and_save(
    *,
    technique_name: str,
    qnode,
    X_train_in: np.ndarray,
    X_val_in: np.ndarray,
    X_test_in: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    q: int,
    L: int,
    entanglement_pattern_label: str,
    feature_dim: int,
    results_csv: str,
    results_json: str,
    preproc_time_s: Optional[float] = None,
    pattern_for_proxy: str,
    topology: str,
    rot_mode: str,
    block_size_for_proxy: int = 2,
    time_ref_X: Optional[np.ndarray] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # encoding time per sample (approx, fast)
    if time_ref_X is None:
        time_ref_X = X_train_in
    enc_time = time_encoding_per_sample(qnode, time_ref_X)

    # encode full splits
    tr = encode_dataset_with_qnode(qnode, X_train_in, out_dim=feature_dim)
    va = encode_dataset_with_qnode(qnode, X_val_in, out_dim=feature_dim)
    te = encode_dataset_with_qnode(qnode, X_test_in, out_dim=feature_dim)

    Xtr_q, Xva_q, Xte_q = tr["Xq"], va["Xq"], te["Xq"]
    n_fail = tr["n_fail"] + va["n_fail"] + te["n_fail"]

    # downstream clf (avoid deprecated multi_class arg)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(Xtr_q, y_train)

    yva = clf.predict(Xva_q)
    yte = clf.predict(Xte_q)
    m_val = _eval_metrics(y_val, yva)
    m_test = _eval_metrics(y_test, yte)

    res = resource_summary(
        qnode,
        x_sample=np.asarray(time_ref_X[0], dtype=float),
        n_qubits=q,
        pattern_for_proxy=pattern_for_proxy,
        topology=topology,
        rot_mode=rot_mode,
        L=L,
        block_size_for_proxy=block_size_for_proxy,
    )

    row = make_checklist_row(
        dataset_name=DATASET_NAME,
        technique_name=technique_name,
        metrics_test=m_test,
        metrics_val=m_val,
        q=q,
        L=L,
        entanglement_pattern=entanglement_pattern_label,
        feature_dim=feature_dim,
        backend=get_backend_name(qnode),
        shots=get_shots(qnode),
        specs_resource=res,
        enc_time_s_per_sample=enc_time,
        preproc_time_s=preproc_time_s,
        n_encode_fail=n_fail,
    )

    if extra_fields:
        row.update(dict(extra_fields))

    append_result_row(row, csv_path=results_csv, json_path=results_json)
    return row

def compute_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes)
    weights = len(y_train) / (n_classes * counts)
    return weights / weights.mean()

def _softmax(z):
    z = z - qml.numpy.max(z, axis=1, keepdims=True)
    e = qml.numpy.exp(z)
    return e / qml.numpy.sum(e, axis=1, keepdims=True)


def run_and_save_vqc(
    *,
    technique_name: str,
    qnode_trainable,
    X_train_in: np.ndarray,
    X_val_in: np.ndarray,
    X_test_in: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    q: int,
    L: int,
    entanglement_pattern_label: str,
    feature_dim: int,
    results_csv: str,
    results_json: str,
    preproc_time_s: Optional[float] = None,
    pattern_for_proxy: str,
    topology: str,
    rot_mode: str,
    block_size_for_proxy: int = 2,
    # VQC training params
    n_classes: int = 3,
    var_layers: int = 1,
    steps: int = 200,
    lr: float = 0.05,
    batch_size: int = 64,
    seed: int = 0,
    use_class_weights: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Minimal VQC training:
      feats_i = qnode_trainable(x_i, w)
      logits  = feats @ W + b
      loss    = cross-entropy
    Trains (w, W, b) using PennyLane AdamOptimizer.
    """
    rng = np.random.default_rng(seed)

    # Trainable circuit params: (var_layers, q, 3)
    w = 0.01 * qml.numpy.array(rng.normal(size=(int(var_layers), int(q), 3)), requires_grad=True)

    # Trainable linear head
    W = 0.01 * qml.numpy.array(rng.normal(size=(int(feature_dim), int(n_classes))), requires_grad=True)
    b = qml.numpy.zeros((int(n_classes),), requires_grad=True)

    opt = qml.optimize.AdamOptimizer(stepsize=float(lr))

    X_train_in = np.asarray(X_train_in, dtype=float)
    X_val_in = np.asarray(X_val_in, dtype=float)
    X_test_in = np.asarray(X_test_in, dtype=float)

    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    if use_class_weights:
        class_weights = compute_class_weights(y_train, n_classes)
        class_weights_qml = qml.numpy.array(class_weights, requires_grad=False)
    else:
        class_weights_qml = qml.numpy.ones(n_classes, requires_grad=False)

    def forward_batch(Xb, wb, Wb, bb):
        feats = []
        for i in range(Xb.shape[0]):
            fi = qml.numpy.asarray(qnode_trainable(Xb[i], wb), dtype=float)
            feats.append(fi)
        F = qml.numpy.stack(feats, axis=0)  # (B, d)
        return F @ Wb + bb

    def loss_fn(wb, Wb, bb, Xb, yb, l2=1e-3):
        logits = forward_batch(Xb, wb, Wb, bb)
        probs = _softmax(logits)
        eps = 1e-12
        idx = qml.numpy.arange(len(yb))
        p = probs[idx, yb]

        sample_losses = -qml.numpy.log(qml.numpy.clip(p, eps, 1.0))
        sample_weights = qml.numpy.array([class_weights_qml[int(y)] for y in yb])
        ce = qml.numpy.mean(sample_losses * sample_weights)

        wd = l2 * qml.numpy.sum(Wb * Wb)  # L2 on W only
        return ce + wd

    best_val = float("inf")
    best_w = None
    best_W = None
    best_b = None
    best_acc = -1.0
    patience = 10  # numero di log senza miglioramenti
    min_delta = 1e-4  # miglioramento minimo richiesto su acc (evita falsi ✓)
    no_improve = 0

    # small fixed val subset for cheap tracking
    val_idx = rng.choice(len(X_val_in), size=min(1024, len(X_val_in)), replace=False)
    Xv = X_val_in[val_idx]
    yv = y_val[val_idx]

    log_every = 10
    t0 = time.time()

    # Training loop
    n = X_train_in.shape[0]
    for step in range(int(steps)):
        idx = rng.choice(n, size=min(int(batch_size), n), replace=False)
        Xb = X_train_in[idx]
        yb = y_train[idx]

        (w, W, b), train_cost = opt.step_and_cost(
            lambda ww, WW, bb: loss_fn(ww, WW, bb, Xb, yb),
            w, W, b
        )

        if (step + 1) % log_every == 0:
            # loss su subset validation
            vcost = float(loss_fn(w, W, b, Xv, yv))

            # accuracy su subset validation (stesso Xv,yv)
            logits_v = forward_batch(Xv, w, W, b)
            pred_v = np.argmax(np.asarray(logits_v), axis=1)
            acc_v = float((pred_v == np.asarray(yv)).mean())

            # criterio di miglioramento su accuracy
            improved = acc_v > (best_acc + min_delta)

            if improved:
                best_acc = acc_v
                best_val = vcost  # opzionale: solo per log
                best_w = qml.numpy.array(w, requires_grad=False)
                best_W = qml.numpy.array(W, requires_grad=False)
                best_b = qml.numpy.array(b, requires_grad=False)
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - t0
            w_norm = float(np.linalg.norm(np.array(w).ravel()))
            W_norm = float(np.linalg.norm(np.array(W).ravel()))
            b_norm = float(np.linalg.norm(np.array(b).ravel()))

            print(
                f"[{step + 1:6d}/{int(steps)}] "
                f"train={float(train_cost):.6f}  val={vcost:.6f}  acc={acc_v:.4f}  "
                f"best_acc={best_acc:.4f}  "
                f"{'✓' if improved else ' '}  "
                f"pat={no_improve}/{patience}  "
                f"||w||={w_norm:.3e} ||W||={W_norm:.3e} ||b||={b_norm:.3e}  "
                f"t={elapsed:.1f}s"
            )

            # EARLY STOPPING
            if no_improve >= patience:
                print(
                    f"Early stopping at step {step + 1}: "
                    f"no val-acc improvement for {patience} logs. "
                    f"Best acc={best_acc:.4f} (val loss at best={best_val:.6f})."
                )
                break

    if best_w is not None:
        w, W, b = best_w, best_W, best_b

    # Prediction helper
    def predict_split(Xs):
        feats = np.zeros((Xs.shape[0], int(feature_dim)), dtype=float)
        for i in range(Xs.shape[0]):
            feats[i] = np.asarray(qnode_trainable(Xs[i], w), dtype=float).ravel()
        logits = feats @ np.asarray(W, dtype=float) + np.asarray(b, dtype=float)
        return np.argmax(logits, axis=1)

    yva = predict_split(X_val_in)
    yte = predict_split(X_test_in)

    m_val = _eval_metrics(y_val, yva)
    m_test = _eval_metrics(y_test, yte)

    # Resource logging (use a sample)
    time_ref_X = X_train_in
    enc_time = time_encoding_per_sample(lambda xx: qnode_trainable(xx, w), time_ref_X)

    res = resource_summary(
        lambda xx: qnode_trainable(xx, w),
        x_sample=np.asarray(time_ref_X[0], dtype=float),
        n_qubits=q,
        pattern_for_proxy=pattern_for_proxy,
        topology=topology,
        rot_mode=rot_mode,
        L=L,
        block_size_for_proxy=block_size_for_proxy,
    )

    row = make_checklist_row(
        dataset_name=DATASET_NAME,
        technique_name=technique_name,
        metrics_test=m_test,
        metrics_val=m_val,
        q=q,
        L=L,
        entanglement_pattern=entanglement_pattern_label,
        feature_dim=int(feature_dim),
        backend=get_backend_name(qnode_trainable),
        shots=get_shots(qnode_trainable),
        specs_resource=res,
        enc_time_s_per_sample=float(enc_time),
        preproc_time_s=preproc_time_s,
        n_encode_fail=0,
    )

    row.update({
        "trainable": True,
        "vqc_var_layers": int(var_layers),
        "vqc_steps": int(steps),
        "vqc_lr": float(lr),
        "vqc_batch": int(batch_size),
    })

    if extra_fields:
        row.update(dict(extra_fields))

    append_result_row(row, csv_path=results_csv, json_path=results_json)
    return row


# =========================
# CLI
# =========================

def build_common_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--outdir", type=str, default="runs", help="Output directory (results + artifacts).")
    p.add_argument("--cache_dir", type=str, default="cache", help="Cache directory for preprocessing arrays.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, default=0, help="0 -> analytic (None), else number of shots.")
    p.add_argument("--device", type=str, default="default.qubit")

    p.add_argument("--n_qubits", type=int, default=8)
    p.add_argument("--entangler", type=str, default="cnot", choices=["cnot", "cz"])
    p.add_argument("--topology", type=str, default="chain", choices=["chain", "ring"])
    p.add_argument("--rot_mode", type=str, default="single", choices=["single", "xyz"])
    p.add_argument("--axis", type=str, default="ry", choices=["rx", "ry", "rz"])

    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)

    p.add_argument("--q_amp", type=int, default=6, help="Amplitude encoding qubits (k=2^q_amp).")
    p.add_argument("--k_base", type=int, default=0, help="Light compression base dim (0 -> auto).")
    p.add_argument("--char_ng_min", type=int, default=3)
    p.add_argument("--char_ng_max", type=int, default=5)

    return p


def cfg_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    shots = None if int(args.shots) == 0 else int(args.shots)
    return BenchmarkConfig(
        random_state=int(args.seed),
        device_name=str(args.device),
        shots=shots,
        n_qubits=int(args.n_qubits),
        entangler=str(args.entangler),  # type: ignore
        topology=str(args.topology),    # type: ignore
        rot_mode=str(args.rot_mode),    # type: ignore
        axis=str(args.axis),            # type: ignore
        ngram_range=(int(args.ngram_min), int(args.ngram_max)),
        min_df=int(args.min_df),
        max_df=float(args.max_df),
        q_amp=int(args.q_amp),
        k_base=int(args.k_base),
        char_ngram_range=(int(args.char_ng_min), int(args.char_ng_max)),
    )
