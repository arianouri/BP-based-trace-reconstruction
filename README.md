# Belief-Propagation-Based Trace Reconstruction over IDS Channels &nbsp; [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18355136.svg)](https://doi.org/10.5281/zenodo.18355136)


This repository considers reconstruction of source sequences from multiple noisy traces affected by insertions, deletions, and substitutions.
- Preprint available at [arXiv:2601.18920 [cs.IT]](https://arxiv.org/abs/2601.18920).  


# Key Files:

*   **`main_trace_reconstruction.py`**: Reconstructs source sequences from multiple noisy copies (traces) provided in `input.txt`
    *   **Input:** `input.txt` (paste your noisy traces here)
        *   Each cluster of traces corresponding to an originating source sequence should be separated by a line of `=` symbols
        *   Source-sequences/Traces should use the alphabet `{A, C, G, T}`
    *   **Variables to Set:**
        *   `N`: Length of the source sequence
        *   `max_ITER`: Maximum iterations (10 recommended for 2-15 traces)
        *   `p_ins`: Insertion probability (approximation)
        *   `p_del`: Deletion probability (approximation)
        *   `p_sub`: Substitution probability (approximation)
    *   **Output:** `output.txt` (estimated source sequences)

*   **`main_single_center_REAL_DNA_DATASET.py`**: Applies the algorithm to a real-world data-set of DNA nanopore sequencing reads from [https://github.com/microsoft/clustered-nanopore-reads-dataset](https://github.com/microsoft/clustered-nanopore-reads-dataset)
	*   This file requires `editdistance` library
    *   **Input:** `Centers.txt` (source sequences) and `Clusters.txt` (noisy traces)
    *   **Variables to Set:**
        *   `line_number_center`: Line number (1-indexed) of a source sequence from `Centers.txt` to recover from its corresponding noisy traces within `Clusters.txt`
        *   `K`: Number of traces to use for reconstruction
        *   `max_ITER`: Maximum iterations (typically `K` for `K > 2`, around `5` for `K=2`)
    *   **Output:** Error rate displayed on the terminal for the specified sequence

*   **`main_iter_over_centers_REAL_DNA_DATASET.py`**: Iteratively runs the algorithm on random sequences from the above-mentioned data-set, displaying the average error rate on the terminal window
	*   This file requires `editdistance` library
    *   **Variables to Set:** `K` and `max_ITER` (similar to `main_single_center_REAL_DNA_DATASET.py`)
    *   **Output:** Iteratively displayed average error rate

*   **`main_RANDOM_DATA.py`**: Runs the algorithm on randomly generated sequences, allowing for control over channel parameters
	*   This file requires `editdistance` library
    *   **Variables to Set:** `K`, `N`, `max_ITER`, `p_ins`, `p_del`, `p_sub`
    *   **Output:** Iteratively displayed average error rate

## Dataset and License
This repository includes the **Clustered Nanopore Reads (CNR)** dataset originally released by Microsoft Corporation in support of *“Trellis BMA: Coded trace reconstruction on IDS channels for DNA storage”* (ISIT 2021). The dataset files (`Centers.txt` and `Clusters.txt`) are distributed under the **MIT License** from the original repository (https://github.com/microsoft/clustered-nanopore-reads-dataset).
