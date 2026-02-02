#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-runs}"
CACHEDIR="${2:-cache}"

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 4
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 4
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 4
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 4

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}'

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 12
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 12
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 12
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 12

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 16
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 16
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 16
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 16


QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 4 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 4 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 4 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 4 --L 2

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --L 2

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 12 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 12 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 12 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 12 --L 2

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 16 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 16 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 16 --L 2
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 16 --L 2


QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 4 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 4 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 4 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 4 --L 3

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --L 3

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 12 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 12 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 12 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 12 --L 3

QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech01_angle_linear.py --n_qubits 16 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech02_angle_full_entanglement.py --n_qubits 16 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech03_amplitude_encoding.py --n_qubits 16 --L 3
QNLP_DEBUG=1 QNLP_DEBUG_EVERY=200 python tech04_word2ket.py --w2k_encoding angle --w2k_cfg_json '{"embedding_dim":8,"order":4,"rank":1,"use_xs":true}' --n_qubits 16 --L 3