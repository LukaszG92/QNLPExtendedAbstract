"""
Common utilities for QNLP experiments with circuit metrics.
Fixed version for PennyLane specs API.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

import numpy as np
import pennylane as qml
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

# Word2Ket dependency
try:
    import word2ket  # type: ignore
except ImportError as e:
    raise ImportError(
        "Missing required dependency 'word2ket'. Install with: pip install word2ket"
    ) from e


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for experiments"""
    random_state: int = 42
    device_name: str = "default.qubit"
    shots: int = 256

    # Circuit parameters
    n_qubits: int = 8
    entangler: Literal["cnot", "cz", "none"] = "cnot"
    topology: Literal["chain", "ring"] = "chain"
    axis: Literal["rx", "ry", "rz"] = "ry"
    L: int = 1

    # Preprocessing
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

    # Classifier
    clf_type: Literal["linearsvc", "ridge"] = "linearsvc"
    n_runs: int = 5


# ============================================================================
# Dataset
# ============================================================================

def load_tweeteval_sentiment() -> Dict[str, np.ndarray]:
    """Load TweetEval sentiment dataset"""
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    return {
        "X_text_train": np.array(ds["train"]["text"], dtype=object),
        "y_train": np.array(ds["train"]["label"], dtype=int),
        "X_text_val": np.array(ds["validation"]["text"], dtype=object),
        "y_val": np.array(ds["validation"]["label"], dtype=int),
        "X_text_test": np.array(ds["test"]["text"], dtype=object),
        "y_test": np.array(ds["test"]["label"], dtype=int),
    }


# ============================================================================
# Preprocessing
# ============================================================================

def tfidf_svd_angles(
    X_text_train: np.ndarray,
    X_text_val: np.ndarray,
    X_text_test: np.ndarray,
    n_components: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """TF-IDF → SVD → angle encoding"""
    t0 = time.perf_counter()

    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    Xtr_tfidf = tfidf.fit_transform(X_text_train)
    Xva_tfidf = tfidf.transform(X_text_val)
    Xte_tfidf = tfidf.transform(X_text_test)

    # SVD
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_train = svd.fit_transform(Xtr_tfidf).astype(np.float32)
    X_val = svd.transform(Xva_tfidf).astype(np.float32)
    X_test = svd.transform(Xte_tfidf).astype(np.float32)

    # Scale to angles [-π, π]
    X_train_ang = (np.pi * np.tanh(X_train)).astype(np.float32)
    X_val_ang = (np.pi * np.tanh(X_val)).astype(np.float32)
    X_test_ang = (np.pi * np.tanh(X_test)).astype(np.float32)

    preproc_time = time.perf_counter() - t0

    return {
        "X_train": X_train_ang,
        "X_val": X_val_ang,
        "X_test": X_test_ang,
        "preproc_time": preproc_time,
    }


def hashing_amp_vectors(
    X_text: np.ndarray,
    k_amp: int,
    random_state: int,
) -> np.ndarray:
    """HashingVectorizer → TF-IDF → L2 normalization for amplitude encoding"""
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
    X_dense = X_tfidf.toarray().astype(np.float64)

    # Normalize
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_dense = X_dense / norms

    # Handle zero rows
    bad = norms.ravel() < 1e-12
    if np.any(bad):
        X_dense[bad, :] = 0.0
        X_dense[bad, 0] = 1.0

    return X_dense.astype(np.float32)


# ============================================================================
# Word2Ket Embedding
# ============================================================================

def build_word2ket_vocabulary(texts: list[str]) -> Dict[str, int]:
    """
    Build vocabulary from training texts ONCE.
    This ensures consistent token IDs across train/val/test splits.

    Args:
        texts: List of training texts

    Returns:
        Dictionary mapping tokens to integer IDs
    """
    token2id = {"<UNK>": 0}
    for text in texts:
        for tok in text.lower().split():
            if tok not in token2id:
                token2id[tok] = len(token2id)

    print(f"  Built Word2Ket vocabulary: {len(token2id)} tokens")
    return token2id


def create_word2ket_embedder(
    vocab_size: int,
    embedding_dim: int,
    order: int = 4,
    rank: int = 1,
    use_xs: bool = True,
):
    """
    Create Word2Ket embedder with specified parameters.

    Args:
        vocab_size: Size of vocabulary (number of unique tokens)
        embedding_dim: Dimension of embeddings (q for angle, 2^q for amplitude)
        order: Order of tensor decomposition
        rank: Rank of tensor decomposition
        use_xs: Whether to use EmbeddingKetXS (faster) or EmbeddingKet

    Returns:
        Word2Ket embedder instance
    """
    EmbedClass = word2ket.EmbeddingKetXS if use_xs else word2ket.EmbeddingKet
    embedder = EmbedClass(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        order=order,
        rank=rank,
    )
    print(f"  Created Word2Ket embedder: dim={embedding_dim}, order={order}, rank={rank}")
    return embedder


def word2ket_embed_texts(
    texts: list[str],
    mode: Literal["angle", "amplitude"],
    q: int,
    token2id: Dict[str, int],
    embedder,
    seed: int = 0,
) -> np.ndarray:
    """
    Embed texts using pre-built Word2Ket vocabulary and embedder.

    CRITICAL: This function now requires a pre-built vocabulary (token2id) and embedder
    to ensure consistency across train/val/test splits. The vocabulary should be built
    ONCE on the training set using build_word2ket_vocabulary().

    Args:
        texts: List of texts to embed
        mode: "angle" or "amplitude" encoding
        q: Number of qubits
        token2id: Pre-built vocabulary (from build_word2ket_vocabulary)
        embedder: Pre-built Word2Ket embedder (from create_word2ket_embedder)
        seed: Random seed for reproducibility

    Returns:
        Embedded vectors (N x q for angle, N x 2^q for amplitude)
    """
    import random
    import torch

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get embedding dimension
    embedding_dim = embedder.embedding_dim

    # Encode texts
    n = len(texts)
    out_raw = np.zeros((n, embedding_dim), dtype=np.float32)

    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            # Use pre-built vocabulary, map unknown tokens to <UNK> (ID=0)
            ids = [token2id.get(tok, 0) for tok in tokens]
            tid = torch.tensor(ids, dtype=torch.long)
            w = embedder(tid)
            s = w.mean(dim=0)
            out_raw[i] = s.cpu().numpy()

    # Convert to circuit input format
    if mode == "angle":
        if out_raw.shape[1] >= q:
            u = out_raw[:, :q]
        else:
            u = np.zeros((n, q), dtype=np.float32)
            u[:, :out_raw.shape[1]] = out_raw
        return (np.tanh(u) * np.pi).astype(np.float32)

    # mode == "amplitude"
    k = 2 ** q
    out = np.zeros((n, k), dtype=np.float32)

    for i in range(n):
        v = out_raw[i].ravel()
        if v.size >= k:
            u = v[:k]
        else:
            u = np.zeros(k)
            u[:v.size] = v

        norm = np.linalg.norm(u)
        if norm < 1e-12:
            u[0] = 1.0
            norm = 1.0
        out[i] = (u / norm).astype(np.float32)

    return out


# ============================================================================
# Quantum Circuits
# ============================================================================

def _entangle_linear(n_qubits: int, entangler: str, topology: str) -> None:
    """Apply linear entanglement pattern"""
    for i in range(n_qubits - 1):
        if entangler == "cnot":
            qml.CNOT(wires=[i, i + 1])
        else:  # cz
            qml.CZ(wires=[i, i + 1])

    if topology == "ring" and n_qubits > 2:
        if entangler == "cnot":
            qml.CNOT(wires=[n_qubits - 1, 0])
        else:
            qml.CZ(wires=[n_qubits - 1, 0])


def _entangle_all_to_all(n_qubits: int, entangler: str) -> None:
    """Apply all-to-all entanglement"""
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if entangler == "cnot":
                qml.CNOT(wires=[i, j])
            else:
                qml.CZ(wires=[i, j])


def make_qnode_angle(
    n_qubits: int,
    entanglement: Literal["linear", "all_to_all", "none"],
    entangler: str = "cnot",
    topology: str = "chain",
    axis: str = "ry",
    L: int = 1,
    device_name: str = "default.qubit",
    shots: int = 256,
):
    """Create angle encoding QNode"""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        x = np.asarray(x, dtype=float).ravel()[:n_qubits]
        x = np.pad(x, (0, max(0, n_qubits - len(x))))

        for _ in range(L):
            # Rotations
            for i in range(n_qubits):
                if axis == "rx":
                    qml.RX(x[i], wires=i)
                elif axis == "ry":
                    qml.RY(x[i], wires=i)
                else:
                    qml.RZ(x[i], wires=i)

            # Entanglement
            if entanglement == "linear":
                _entangle_linear(n_qubits, entangler, topology)
            elif entanglement == "all_to_all":
                _entangle_all_to_all(n_qubits, entangler)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def make_qnode_amplitude(
    n_qubits: int,
    L: int = 1,
    entangler: str = "none",
    topology: str = "chain",
    device_name: str = "default.qubit",
    shots: int = 256,
):
    """Create amplitude encoding QNode"""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        x = np.asarray(x, dtype=float)

        # Normalize
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            x = np.zeros_like(x)
            x[0] = 1.0
        else:
            x = x / norm

        for _ in range(L):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=False)

            if entangler != "none":
                _entangle_linear(n_qubits, entangler, topology)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ============================================================================
# Circuit Metrics
# ============================================================================

def compute_circuit_metrics(qnode, x_sample: np.ndarray, n_qubits: int) -> Dict[str, Any]:
    """
    Compute circuit metrics: depth, gate counts, etc.
    Fixed for newer PennyLane API.
    """
    try:
        specs = qml.specs(qnode)(x_sample)

        # Handle both old and new API
        if hasattr(specs, 'resources'):
            # New API: specs is an object
            resources = specs.resources

            # Extract from Resources object
            if hasattr(resources, 'depth'):
                depth = resources.depth
            else:
                depth = None

            if hasattr(resources, 'num_gates'):
                num_gates = resources.num_gates
            else:
                num_gates = None

            if hasattr(resources, 'gate_sizes'):
                gate_sizes = resources.gate_sizes
            else:
                gate_sizes = {}

            if hasattr(resources, 'gate_types'):
                gate_types = resources.gate_types
            else:
                gate_types = {}

        elif isinstance(specs, dict):
            # Old API: specs is a dict
            resources = specs.get("resources", {})
            depth = resources.get("depth", None)
            num_gates = resources.get("num_gates", None)
            gate_sizes = resources.get("gate_sizes", {})
            gate_types = resources.get("gate_types", {})
        else:
            # Fallback
            depth = None
            num_gates = None
            gate_sizes = {}
            gate_types = {}

        # Count gates
        if isinstance(gate_sizes, dict):
            two_qubit_gates = gate_sizes.get(2, 0)
            single_qubit_gates = gate_sizes.get(1, 0)
        else:
            two_qubit_gates = 0
            single_qubit_gates = 0

        if isinstance(gate_types, dict):
            cnot_count = gate_types.get("CNOT", 0)
            cz_count = gate_types.get("CZ", 0)
        else:
            cnot_count = 0
            cz_count = 0

        return {
            "depth": depth,
            "num_gates": num_gates,
            "single_qubit_gates": single_qubit_gates,
            "two_qubit_gates": two_qubit_gates,
            "cnot_gates": cnot_count,
            "cz_gates": cz_count,
            "num_wires": n_qubits,
        }

    except Exception as e:
        print(f"Warning: Could not compute circuit metrics: {e}")
        # Return None values but continue execution
        return {
            "depth": None,
            "num_gates": None,
            "single_qubit_gates": None,
            "two_qubit_gates": None,
            "cnot_gates": None,
            "cz_gates": None,
            "num_wires": n_qubits,
        }


# ============================================================================
# Encoding & Timing
# ============================================================================

def encode_dataset(qnode, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Encode dataset using quantum circuit, return encoded data and time per sample"""
    n = X.shape[0]

    # Get output dimension
    y0 = np.asarray(qnode(X[0]), dtype=float).ravel()
    d = len(y0)

    # Encode all samples with timing
    out = np.zeros((n, d), dtype=np.float32)
    out[0] = y0.astype(np.float32)

    t0 = time.perf_counter()
    for i in range(1, min(n, 100)):  # Time first 100 samples
        yi = np.asarray(qnode(X[i]), dtype=float).ravel()
        out[i] = yi.astype(np.float32)
    t_batch = time.perf_counter() - t0
    time_per_sample = t_batch / 99 if n > 1 else 0.0

    # Encode remaining samples
    for i in range(100, n):
        yi = np.asarray(qnode(X[i]), dtype=float).ravel()
        out[i] = yi.astype(np.float32)

    return out, time_per_sample


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_technique(
    technique_name: str,
    qnode,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    preproc_time: float,
    cfg: ExperimentConfig,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single technique with complete metrics.

    Returns dict with:
    - Accuracy and F1 scores (mean ± std over n_runs)
    - Circuit metrics (depth, gates, etc.)
    - Timing information
    - All configuration parameters
    """
    print(f"\nEvaluating: {technique_name}")
    print("=" * 60)

    # Compute circuit metrics
    print("Computing circuit metrics...")
    circuit_metrics = compute_circuit_metrics(qnode, X_train[0], cfg.n_qubits)
    print(f"  Depth: {circuit_metrics['depth']}")
    print(f"  Total gates: {circuit_metrics['num_gates']}")
    print(f"  Two-qubit gates: {circuit_metrics['two_qubit_gates']}")

    # Multiple runs for statistical significance
    acc_val_list, f1_val_list = [], []
    acc_test_list, f1_test_list = [], []
    encoding_times = []

    # Encode (with timing)
    Xtr_enc, enc_time_tr = encode_dataset(qnode, X_train)
    Xva_enc, _ = encode_dataset(qnode, X_val)
    Xte_enc, _ = encode_dataset(qnode, X_test)
    encoding_times.append(enc_time_tr)

    print(f"\nRunning {cfg.n_runs} evaluations...")
    for r in range(cfg.n_runs):
        # Set seed for this run
        np.random.seed(cfg.random_state + r)

        # Train classifier
        if cfg.clf_type == "linearsvc":
            clf = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced")
        else:
            clf = RidgeClassifier(alpha=1.0)

        clf.fit(Xtr_enc, y_train)

        # Predict
        y_val_pred = clf.predict(Xva_enc)
        y_test_pred = clf.predict(Xte_enc)

        # Metrics
        acc_val_list.append(accuracy_score(y_val, y_val_pred))
        f1_val_list.append(f1_score(y_val, y_val_pred, average="macro"))
        acc_test_list.append(accuracy_score(y_test, y_test_pred))
        f1_test_list.append(f1_score(y_test, y_test_pred, average="macro"))

        print(f"  Run {r+1}/{cfg.n_runs}: "
              f"val_acc={acc_val_list[-1]:.4f}, "
              f"test_acc={acc_test_list[-1]:.4f}, "
              f"test_f1={f1_test_list[-1]:.4f}")

    # Aggregate results
    result = {
        # Identification
        "technique": technique_name,
        "n_qubits": cfg.n_qubits,
        "L": cfg.L,
        "shots": cfg.shots,
        "clf_type": cfg.clf_type,
        "n_runs": cfg.n_runs,

        # Circuit metrics
        "depth": circuit_metrics["depth"],
        "num_gates": circuit_metrics["num_gates"],
        "single_qubit_gates": circuit_metrics["single_qubit_gates"],
        "two_qubit_gates": circuit_metrics["two_qubit_gates"],
        "cnot_gates": circuit_metrics["cnot_gates"],
        "cz_gates": circuit_metrics["cz_gates"],

        # Performance metrics
        "test_acc_mean": float(np.mean(acc_test_list)),
        "test_acc_std": float(np.std(acc_test_list)),
        "test_f1_mean": float(np.mean(f1_test_list)),
        "test_f1_std": float(np.std(f1_test_list)),
        "val_acc_mean": float(np.mean(acc_val_list)),
        "val_acc_std": float(np.std(acc_val_list)),
        "val_f1_mean": float(np.mean(f1_val_list)),
        "val_f1_std": float(np.std(f1_val_list)),

        # Timing
        "preproc_time_s": preproc_time,
        "encoding_time_per_sample_s": float(np.mean(encoding_times)),
        "encoding_time_std_s": float(np.std(encoding_times)),
    }

    # Add extra parameters if provided
    if extra_params:
        result.update(extra_params)

    print(f"\n✓ Final Results:")
    print(f"  Test Accuracy: {result['test_acc_mean']:.4f} ± {result['test_acc_std']:.4f}")
    print(f"  Test F1:       {result['test_f1_mean']:.4f} ± {result['test_f1_std']:.4f}")

    return result


# ============================================================================
# Results I/O
# ============================================================================

def save_result(result: Dict[str, Any], output_file: str) -> None:
    """Append result to CSV file (creates if doesn't exist)"""
    df_new = pd.DataFrame([result])

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(output_file, index=False)
    print(f"\n✓ Results appended to: {output_file}")