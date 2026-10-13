"""Synthetic inputs for the original Chreode evaluator's reference results."""

import pickle

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_case(directory, prediction_format, matrix_kind, control_dtype="float64"):
    directory.mkdir(parents=True, exist_ok=True)
    genes = [f"g{i}" for i in range(32)]
    index = np.arange(32, dtype=np.float64)
    ctrl = np.stack([0.25 + index / 7, 1.125 + index / 5, 0.5 + index / 3]).astype(control_dtype)
    mean = ctrl.mean(axis=0)
    # Saved test truth represents selected cells, rather than the full H5AD pool.
    paired = np.stack([mean + 0.2 + index / 11, mean - 1.3 + index / 13,
                       mean + 0.8 - index / 17, mean]).astype(np.float32)
    pred = (paired + np.stack([np.sin(index) / 7, np.cos(index) / 8,
                               np.sin(index / 3) / 9, np.zeros(32)])).astype(np.float32)
    pred[:, 24:] += 50  # errors outside the shared vocabulary must not affect DE20
    conditions = np.asarray(["a+ctrl", "a+b", "a+b", "ctrl"])
    real = np.vstack([ctrl, mean + 20 + index, mean - 30 - index]).astype(control_dtype)
    matrix = sparse.csr_matrix(real) if matrix_kind == "sparse" else real
    adata_path = directory / "perturb_processed.h5ad"
    ad.AnnData(X=matrix, obs=pd.DataFrame({"condition": ["ctrl"] * 3 + ["a+ctrl", "a+b"]},
                                        index=[f"cell{i}" for i in range(5)]),
               var=pd.DataFrame({"gene_name": genes}, index=[f"ENSG{i}" for i in range(32)])).write_h5ad(adata_path)
    vocab = directory / "shared_vocab_genes.txt"
    vocab.write_text("\n".join(genes[:24]) + "\n")
    prediction = directory / ("predictions.npz" if prediction_format == "npz" else "test_res.pkl")
    if prediction_format == "npz":
        np.savez(prediction, pred=pred[:, ::-1], truth=paired[:, ::-1], conditions=conditions,
                 gene_names=np.asarray([gene.upper() for gene in genes[::-1]]))
    else:
        with prediction.open("wb") as handle:
            pickle.dump({"pred": pred, "truth": paired, "pert_cat": conditions}, handle)
    return prediction, adata_path, vocab
