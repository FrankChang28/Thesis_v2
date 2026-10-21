from __future__ import annotations

import numpy as np
import pandas as pd

from subject_nirs.stage2.comparison import (
    build_heterogeneous_benefit_detail,
    build_final_experiment_table,
    format_final_experiment_table,
    heterogeneous_benefit_missing_cells,
    holm_adjust,
    paired_comparisons,
    paired_rank_biserial,
    plot_heterogeneous_benefit,
    resolve_heterogeneous_benefit_method,
    summarize_heterogeneous_benefit,
)


def _frame() -> pd.DataFrame:
    rows = []
    for subject_id, baseline_rmse, experiment_rmse, bias in (
        (1, 3.0, 2.0, -0.2),
        (2, 4.0, 3.5, 0.1),
        (3, 5.0, 5.5, 0.3),
        (4, 6.0, 4.0, -0.1),
    ):
        rows.append(
            {
                "target": "hc", "experiment": "DRS-only baseline", "training_mode": "baseline",
                "feature_source": "baseline", "fusion_method": "none",
                "training_strategy": "scratch", "structure_control": "none",
                "subject_id": subject_id, "RMSE": baseline_rmse, "MAE": baseline_rmse - 0.2,
                "Bias": bias, "AbsBias": abs(bias), "ErrorStd": baseline_rmse,
            }
        )
        rows.append(
            {
                "target": "hc", "experiment": "DTOF descriptor | Gated residual | Fine-tuning",
                "training_mode": "residual", "feature_source": "latent",
                "fusion_method": "residual", "training_strategy": "finetune",
                "structure_control": "actual", "subject_id": subject_id,
                "RMSE": experiment_rmse, "MAE": experiment_rmse - 0.2,
                "Bias": bias / 2, "AbsBias": abs(bias / 2), "ErrorStd": experiment_rmse,
            }
        )
    return pd.DataFrame(rows)


def _complete_heterogeneity_frame(methods=("residual",)) -> pd.DataFrame:
    rows = []
    for target_index, target in enumerate(("hc", "sto2")):
        for subject_id in range(8):
            baseline_rmse = 1.0 + target_index + subject_id
            common = {
                "target": target,
                "subject_id": subject_id,
                "MAE": baseline_rmse - 0.1,
                "Bias": 0.0,
                "AbsBias": 0.0,
                "ErrorStd": baseline_rmse,
            }
            rows.append(
                {
                    **common,
                    "experiment": "DRS-only baseline",
                    "training_mode": "baseline",
                    "feature_source": "baseline",
                    "fusion_method": "none",
                    "training_strategy": "scratch",
                    "structure_control": "none",
                    "RMSE": baseline_rmse,
                }
            )
            for method in methods:
                for strategy_index, strategy in enumerate(("scratch", "finetune")):
                    for source_index, source in enumerate(("latent", "metadata")):
                        delta = 0.2 * source_index - 0.1 * strategy_index - 0.05 * subject_id
                        label = f"{source} | {method} | {strategy}"
                        rows.append(
                            {
                                **common,
                                "experiment": label,
                                "training_mode": method,
                                "feature_source": source,
                                "fusion_method": method,
                                "training_strategy": strategy,
                                "structure_control": "actual",
                                "RMSE": baseline_rmse + delta,
                            }
                        )
    return pd.DataFrame(rows)


def test_rank_biserial_orientation() -> None:
    assert paired_rank_biserial(np.array([-3.0, -2.0, -1.0])) == 1.0
    assert paired_rank_biserial(np.array([3.0, 2.0, 1.0])) == -1.0


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust(pd.Series([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted.to_numpy(), [0.03, 0.06, 0.06])


def test_final_table_contains_requested_statistics() -> None:
    frame = _frame()
    detail, paired = paired_comparisons(frame, "DRS-only baseline")
    table = build_final_experiment_table(frame, paired)
    formatted = format_final_experiment_table(table)
    fusion = table[table["feature_source"] == "latent"].iloc[0]

    assert len(detail) == 4
    assert fusion["n_paired_subjects"] == 4
    assert fusion["delta_rmse_median"] == -0.75
    assert fusion["delta_rmse_q1"] == -1.25
    assert fusion["delta_rmse_q3"] == -0.25
    assert fusion["improved_subjects_pct"] == 75.0
    assert "Subject RMSE median [Q25, Q75]" in formatted.columns
    assert "Pooled RMSE" in formatted.columns
    assert "Q4 mean ΔRMSE" in formatted.columns
    assert "Holm-adjusted p" not in formatted.columns
    assert "wilcoxon_p_holm" in table.columns


def test_heterogeneity_matrix_uses_common_baseline_quartiles() -> None:
    frame = _complete_heterogeneity_frame()
    paired_detail, _ = paired_comparisons(frame, "DRS-only baseline")
    detail = build_heterogeneous_benefit_detail(frame, paired_detail, "residual")
    summary = summarize_heterogeneous_benefit(detail)

    assert heterogeneous_benefit_missing_cells(frame, "residual") == []
    assert len(detail) == 2 * 2 * 2 * 8
    assert len(summary) == 2 * 2 * 2 * 4
    assert tuple(summary.iloc[0][["target", "training_strategy", "feature_source"]]) == (
        "hc", "scratch", "latent"
    )
    assert (
        detail.groupby(["target", "subject_id"])["difficulty_quartile"].nunique()
        == 1
    ).all()
    first = detail[
        (detail["target"] == "hc")
        & (detail["training_strategy"] == "scratch")
        & (detail["feature_source"] == "latent")
        & (detail["subject_id"] == 0)
    ].iloc[0]
    assert first["delta_RMSE"] == 0.0


def test_heterogeneity_fallback_uses_one_complete_method() -> None:
    frame = _complete_heterogeneity_frame(methods=("residual", "concatenation"))
    incomplete = frame[
        ~(
            (frame["fusion_method"] == "residual")
            & (frame["target"] == "sto2")
            & (frame["training_strategy"] == "scratch")
            & (frame["feature_source"] == "metadata")
        )
    ].copy()
    selected, is_preview, missing = resolve_heterogeneous_benefit_method(
        incomplete,
        requested_method="residual",
        fallback_method="concatenation",
    )

    assert selected == "concatenation"
    assert is_preview
    assert "sto2/scratch/metadata" in missing


def test_heterogeneity_plot_is_created(tmp_path) -> None:
    frame = _complete_heterogeneity_frame()
    paired_detail, _ = paired_comparisons(frame, "DRS-only baseline")
    detail = build_heterogeneous_benefit_detail(frame, paired_detail, "residual")
    output = plot_heterogeneous_benefit(
        detail,
        output_dir=tmp_path,
        fusion_method="residual",
        dpi=72,
    )
    assert output.is_file()


if __name__ == "__main__":
    test_rank_biserial_orientation()
    test_holm_adjustment_is_monotone()
    test_final_table_contains_requested_statistics()
    test_heterogeneity_matrix_uses_common_baseline_quartiles()
    test_heterogeneity_fallback_uses_one_complete_method()
    print("stage2 comparison tests passed")


def test_performance_table_weights_rows_and_keeps_subject_pairing():
    frame = _frame()
    frame["n_test"] = frame["subject_id"].map({1: 10, 2: 20, 3: 30, 4: 40})
    frame.loc[(frame["feature_source"] == "latent") & (frame["subject_id"] == 3), "RMSE"] = 6.5
    _, paired = paired_comparisons(frame, "DRS-only baseline")
    table = build_final_experiment_table(frame, paired)
    baseline = table[table["feature_source"] == "baseline"].iloc[0]
    fusion = table[table["feature_source"] == "latent"].iloc[0]
    np.testing.assert_allclose(baseline.pooled_rmse, np.sqrt(26.0))
    np.testing.assert_allclose(fusion.pooled_rmse, np.sqrt(21.925))
    assert fusion.delta_rmse_mean == -0.5
    assert fusion.delta_rmse_median == -0.75
    np.testing.assert_allclose(fusion.mean_absolute_subject_bias, 0.0875)
    assert fusion.q1_mean_delta_rmse == -1.0
    assert fusion.q2_mean_delta_rmse == -0.5
    assert fusion.q3_mean_delta_rmse == 1.5
    assert fusion.q4_mean_delta_rmse == -2.0
    assert all(fusion[f"q{i}_n_subjects"] == 1 for i in range(1, 5))
    assert fusion.delta_rmse_mean_ci_low <= -0.5 <= fusion.delta_rmse_mean_ci_high
    assert np.isnan(baseline.delta_rmse_mean)
    repeated = build_final_experiment_table(frame, paired)
    pd.testing.assert_frame_equal(table, repeated)


def test_missing_test_counts_do_not_invent_a_pooled_score():
    frame = _frame()
    _, paired = paired_comparisons(frame, "DRS-only baseline")
    table = build_final_experiment_table(frame, paired)
    assert table.pooled_rmse.isna().all()
    assert format_final_experiment_table(table)["Pooled RMSE"].eq("—").all()


def test_baseline_only_table_has_no_paired_claims():
    frame = _frame().query("feature_source == 'baseline'").copy()
    frame["n_test"] = 10
    _, paired = paired_comparisons(frame, "DRS-only baseline")
    table = build_final_experiment_table(frame, paired)
    formatted = format_final_experiment_table(table)
    assert len(table) == 1
    assert table.delta_rmse_mean.isna().all()
    assert formatted["Mean paired ΔRMSE [95% CI]"].eq("—").all()
    assert formatted["Q4 mean ΔRMSE"].eq("—").all()
