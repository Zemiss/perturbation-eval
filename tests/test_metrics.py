import numpy as np
import pytest

from evaluate.metrics import compute_metrics


def score(pred, truth, control=None, **kwargs):
    truth = np.asarray(truth, dtype=float)
    return compute_metrics(
        np.asarray(pred, dtype=float), truth,
        np.zeros_like(truth) if control is None else np.asarray(control, dtype=float),
        [f"g{i}" for i in range(truth.shape[1])],
        [f"p{i}" for i in range(truth.shape[0])], **kwargs,
    )


def test_perfect_predictions():
    result = score([[1, 2, 3], [6, 4, 2]], [[1, 2, 3], [6, 4, 2]])
    assert result["metrics"] == pytest.approx(
        {"de20_mse": 0, "de20_delta_r": 1, "pearson_delta": 1, "pds": 1}
    )


def test_top_de_genes_come_only_from_truth_and_mse_is_macro_mean():
    result = score([[2, 2, 100], [9, 5, 100]], [[3, 2, 1], [6, 4, 2]], top_k=2)
    assert result["metrics"]["de20_mse"] == pytest.approx((0.5 + 5) / 2)
    assert result["per_condition"][0]["de_genes"] == ["g0", "g1"]
    assert result["per_condition"][1]["de_genes"] == ["g0", "g1"]


def test_signed_de_correlation_and_absolute_global_correlation_differ():
    result = score([[-1, -2, -4]], [[1, 2, 4]])
    assert result["metrics"]["de20_delta_r"] == pytest.approx(-1)
    assert result["metrics"]["pearson_delta"] == pytest.approx(1)


def test_same_truth_control_is_used_for_both_deltas():
    result = score([[11, 22, 43]], [[9, 18, 37]], [[10, 20, 40]])
    assert result["metrics"]["de20_delta_r"] == pytest.approx(-1)
    assert result["metrics"]["pearson_delta"] == pytest.approx(1)


def test_pearson_delta_is_flattened_not_mean_per_perturbation():
    pred = np.array([[1, 2, 3], [30, 20, 10]], dtype=float)
    truth = np.array([[1, 2, 3], [10, 20, 30]], dtype=float)
    result = score(pred, truth)
    assert result["metrics"]["pearson_delta"] == pytest.approx(
        np.corrcoef(truth.ravel(), pred.ravel())[0, 1]
    )
    assert result["metrics"]["de20_delta_r"] == pytest.approx(0)


def test_pds_pdf_normalization_and_strict_ties():
    result = score([[5, 0], [10, 0]], [[0, 0], [10, 0]])
    assert result["metrics"]["pds"] == 1  # first prediction ties both references
    swapped = score([[10, 0], [0, 0]], [[0, 0], [10, 0]])
    assert swapped["metrics"]["pds"] == 0  # 1 - 2*(1+1)/2**2


def test_pds_does_not_compare_across_background_groups():
    result = score(
        [[0, 1], [10, 1], [100, 1]], [[0, 1], [10, 1], [100, 1]],
        groups=[("a",), ("a",), ("b",)],
    )
    assert result["metrics"]["pds"] == 1
    assert [r["pds_candidates"] for r in result["per_condition"]] == [2, 2, 1]


def test_constant_correlations_are_zero_and_reported():
    result = score([[1, 1]], [[1, 1]])
    assert result["metrics"]["de20_delta_r"] == 0
    assert result["metrics"]["pearson_delta"] == 0
    assert result["undefined_correlations"] == {"de20_delta_r": 1, "pearson_delta": 1}


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_values_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        score([[bad, 1]], [[0, 1]])


def test_less_than_twenty_genes_reports_actual_k():
    result = score([[1, 2]], [[1, 2]])
    assert result["per_condition"][0]["de_k"] == 2


def test_default_twenty_excludes_errors_outside_top_de_genes():
    truth = np.arange(32, dtype=float)[None, :]
    pred = truth.copy()
    pred[0, :12] = 1000
    result = score(pred, truth)
    assert result["per_condition"][0]["de_k"] == 20
    assert result["per_condition"][0]["de_genes"] == [f"g{i}" for i in range(31, 11, -1)]
    assert result["metrics"]["de20_mse"] == 0


def test_chreode_de20_ties_follow_original_argsort():
    truth = np.ones((1, 32))
    pred = np.arange(32, dtype=float)[None, :]
    result = score(pred, truth, de_protocol="chreode")
    chosen = np.argsort(np.abs(truth[0]))[-20:][::-1]
    assert result["per_condition"][0]["de_genes"] == [f"g{i}" for i in chosen]
    assert result["metrics"]["de20_mse"] == pytest.approx(np.mean((pred[0, chosen] - 1) ** 2))
