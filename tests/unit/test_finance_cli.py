"""Exhaustive unit test suite for fuzzymatch_finance.cli achieving 100.00% statement coverage."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import openpyxl
import pandas as pd
import pytest

from fuzzymatch_finance.cli import (
    FinanceMatchResults,
    _load_tabular_data,
    auto_match_finance,
    main,
)


def test_cli_no_args_returns_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert (
        "usage:" in captured.out.lower()
        or "usage:" in captured.err.lower()
        or "match" in captured.out
    )


def test_cli_help_flag() -> None:
    exit_code = main(["--help"])
    assert exit_code == 0


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--version"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "fuzzymatch-finance" in captured.out


def test_cli_match_help() -> None:
    exit_code = main(["match", "--help"])
    assert exit_code == 0


def test_cli_unknown_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["invalid_subcommand"])
    assert exit_code == 2


def test_cli_match_missing_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["match"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_match_missing_one_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f1 = tmp_path / "left.csv"
    pd.DataFrame({"id": [1]}).to_csv(f1, index=False)
    exit_code = main(["match", str(f1)])
    assert exit_code == 2


def test_cli_file_not_found_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f1 = tmp_path / "valid.csv"
    pd.DataFrame({"id": [1]}).to_csv(f1, index=False)
    exit_code = main(["match", str(f1), str(tmp_path / "missing.csv")])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_cli_corrupted_file_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad_f = tmp_path / "bad.csv"
    bad_f.write_bytes(b"\x00\xff\xfe\x01\x02\x03\x04\x05")
    good_f = tmp_path / "good.csv"
    pd.DataFrame({"id": [1]}).to_csv(good_f, index=False)

    exit_code = main(["match", str(bad_f), str(good_f)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_load_tabular_data_variations(tmp_path: Path) -> None:
    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_text("id\tname\n1\tAlpha\n")
    df_tsv = _load_tabular_data(str(tsv_file))
    assert len(df_tsv) == 1

    bad_excel = tmp_path / "corrupt.xlsx"
    bad_excel.write_bytes(b"not an excel file")
    with pytest.raises(ValueError, match="Failed to parse Excel file"):
        _load_tabular_data(str(bad_excel))

    other_ext = tmp_path / "data.dat"
    other_ext.write_text("id,name\n1,Alpha\n")
    df_other = _load_tabular_data(str(other_ext))
    assert len(df_other) == 1

    bad_other = tmp_path / "bad.dat"
    bad_other.write_bytes(b"\x00\xff\xfe\x01\x02")
    with pytest.raises(ValueError, match="Unsupported file format"):
        _load_tabular_data(str(bad_other))


def test_cli_positional_csv_matching(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f_l = tmp_path / "bank.csv"
    f_r = tmp_path / "ap.csv"
    out_csv = tmp_path / "reconciled.csv"

    df_l = pd.DataFrame(
        {
            "bank_id": ["B1", "B2"],
            "statement_description": ["Acme Industrial Inc", "Summit Chemical LLC"],
            "cleared_amount": [1200.50, 450.00],
            "cleared_date": ["2025-01-10", "2025-01-15"],
        }
    )
    df_r = pd.DataFrame(
        {
            "ledger_id": ["L1", "L2"],
            "payee_name": ["Acme Industrial", "Summit Chem Co"],
            "invoice_amount": [1200.00, 450.00],
            "invoice_date": ["2025-01-11", "2025-01-14"],
        }
    )
    df_l.to_csv(f_l, index=False)
    df_r.to_csv(f_r, index=False)

    exit_code = main(["match", str(f_l), str(f_r), "--output", str(out_csv)])
    assert exit_code == 0
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert len(out_df) == 2
    captured = capsys.readouterr()
    assert "reconciled" in captured.out.lower()
    assert "average match confidence" in captured.out.lower()


def test_cli_flags_excel_and_review_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f_l = tmp_path / "bank.xlsx"
    f_r = tmp_path / "ap.xlsx"
    out_xlsx = tmp_path / "reconciled.xlsx"
    review_xlsx = tmp_path / "curation.xlsx"

    df_l = pd.DataFrame(
        {
            "id": ["B1"],
            "name": ["Beacon Global Corp"],
            "amount": [999.00],
            "date": ["2025-03-01"],
        }
    )
    df_r = pd.DataFrame(
        {
            "id": ["L1"],
            "name": ["Beacon Global"],
            "amount": [999.00],
            "date": ["2025-03-01"],
        }
    )
    df_l.to_excel(f_l, index=False)
    df_r.to_excel(f_r, index=False)

    exit_code = main(
        [
            "match",
            "--left",
            str(f_l),
            "--right",
            str(f_r),
            "--output",
            str(out_xlsx),
            "--review-output",
            str(review_xlsx),
            "--explain",
        ]
    )
    assert exit_code == 0
    assert out_xlsx.exists()
    assert review_xlsx.exists()

    wb = openpyxl.load_workbook(review_xlsx)
    ws = wb["review"]
    headers = [cell.value for cell in ws[1]]
    assert "decision" in headers
    assert "fuzzy_score" in headers
    assert len(ws.data_validations.dataValidation) >= 1

    captured = capsys.readouterr()
    assert "explanations" in captured.out.lower()


def test_cli_review_out_alias(tmp_path: Path) -> None:
    f_l = tmp_path / "l.csv"
    f_r = tmp_path / "r.csv"
    review_xlsx = tmp_path / "alias_curation.xlsx"

    pd.DataFrame({"id": ["A"], "name": ["Alpha"]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": ["B"], "name": ["Alpha"]}).to_csv(f_r, index=False)

    exit_code = main(["match", str(f_l), str(f_r), "--review-out", str(review_xlsx)])
    assert exit_code == 0
    assert review_xlsx.exists()


def test_cli_explicit_column_mappings_and_tolerances(tmp_path: Path) -> None:
    f_l = tmp_path / "left.csv"
    f_r = tmp_path / "right.csv"

    df_l = pd.DataFrame(
        {
            "my_id_l": ["X1"],
            "my_txt_l": ["Delta Logistics"],
            "my_amt_l": [100.00],
            "my_dt_l": ["2025-05-01"],
        }
    )
    df_r = pd.DataFrame(
        {
            "my_id_r": ["Y1"],
            "my_txt_r": ["Delta Logistics Inc"],
            "my_amt_r": [101.50],
            "my_dt_r": ["2025-05-04"],
        }
    )
    df_l.to_csv(f_l, index=False)
    df_r.to_csv(f_r, index=False)

    exit_code = main(
        [
            "match",
            str(f_l),
            str(f_r),
            "--id-col-left",
            "my_id_l",
            "--id-col-right",
            "my_id_r",
            "--text-col-left",
            "my_txt_l",
            "--text-col-right",
            "my_txt_r",
            "--amount-col-left",
            "my_amt_l",
            "--amount-col-right",
            "my_amt_r",
            "--date-col-left",
            "my_dt_l",
            "--date-col-right",
            "my_dt_r",
            "--epsilon",
            "2.0",
            "--delta-days",
            "5.0",
            "--threshold",
            "0.6",
            "--budget-cap",
            "1000",
            "--gated",
        ]
    )
    assert exit_code == 0


def test_cli_invalid_explicit_column_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f_l = tmp_path / "l.csv"
    f_r = tmp_path / "r.csv"
    pd.DataFrame({"id": [1], "name": ["A"]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": [2], "name": ["B"]}).to_csv(f_r, index=False)

    exit_code = main(["match", str(f_l), str(f_r), "--amount-col-left", "non_existent_column"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_cli_empty_tables_handling(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f_l = tmp_path / "empty_l.csv"
    f_r = tmp_path / "empty_r.csv"
    pd.DataFrame(columns=["id", "name", "amount"]).to_csv(f_l, index=False)
    pd.DataFrame(columns=["id", "name", "amount"]).to_csv(f_r, index=False)

    exit_code = main(["match", str(f_l), str(f_r)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "0 records" in captured.out


def test_cli_tables_without_amount_or_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f_l = tmp_path / "text_only_l.csv"
    f_r = tmp_path / "text_only_r.csv"
    pd.DataFrame({"vendor_id": ["V1"], "vendor_name": ["Acme"]}).to_csv(f_l, index=False)
    pd.DataFrame({"ap_id": ["A1"], "vendor_name": ["Acme LLC"]}).to_csv(f_r, index=False)

    exit_code = main(["match", str(f_l), str(f_r)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Volume: N/A" in captured.out


def test_auto_match_finance_python_api() -> None:
    df_l = pd.DataFrame({"id": [1], "name": ["Acme Corp"], "amount": [100.0]})
    df_r = pd.DataFrame({"id": [2], "name": ["Acme"], "amount": [100.0]})

    results = auto_match_finance(df_l, df_r)
    assert isinstance(results, FinanceMatchResults)
    assert len(results.matches) == 1
    assert len(results.pairs) == 1
    assert len(results.clusters) == 1
    assert results.summary.reconciled_count == 1
    assert isinstance(results.to_dataframe(), pd.DataFrame)


def test_auto_match_finance_empty_export(tmp_path: Path) -> None:
    results = auto_match_finance(pd.DataFrame(), pd.DataFrame())
    out_xlsx = tmp_path / "empty_review.xlsx"
    results.export_review_xlsx(out_xlsx)
    assert out_xlsx.exists()


def test_auto_match_finance_large_and_adaptive_weights() -> None:
    df_l = pd.DataFrame(
        {
            "id": range(100),
            "name": [f"Vendor {i}" for i in range(100)],
            "date": ["2025-01-01"] * 100,
        }
    )
    df_r = pd.DataFrame(
        {
            "id": range(100),
            "name": [f"Vendor {i}" for i in range(100)],
            "date": ["2025-01-01"] * 100,
        }
    )
    results = auto_match_finance(df_l, df_r, budget_cap=500)
    assert len(results.matches) > 0


def test_auto_match_finance_date_and_amount_exceptions() -> None:
    df_l = pd.DataFrame(
        {
            "name": ["A"],
            "amount": ["invalid_amt"],
            "date": ["invalid_date"],
        }
    )
    df_r = pd.DataFrame(
        {
            "name": ["A"],
            "amount": ["invalid_amt_2"],
            "date": ["invalid_date_2"],
        }
    )
    results = auto_match_finance(df_l, df_r, threshold=0.5)
    assert len(results.matches) == 1


def test_auto_match_finance_no_passing_pairs() -> None:
    df_l = pd.DataFrame({"id": [1], "name": ["Alpha Corp"]})
    df_r = pd.DataFrame({"id": [2], "name": ["Zeta Global"]})
    results = auto_match_finance(df_l, df_r, threshold=0.99)
    assert len(results.matches) == 0
    assert len(results.clusters) == 0


def test_cli_main_as_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f_l = tmp_path / "left.csv"
    f_r = tmp_path / "right.csv"
    pd.DataFrame({"id": [1], "name": ["Test"]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": [2], "name": ["Test"]}).to_csv(f_r, index=False)

    monkeypatch.setattr(sys, "argv", ["fuzzymatch-finance", "match", str(f_l), str(f_r)])
    exit_code = main()
    assert exit_code == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_runpy_entrypoint() -> None:
    with patch("sys.argv", ["fuzzymatch-finance", "--help"]):
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("fuzzymatch_finance.cli", run_name="__main__")
        assert exc.value.code == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_runpy_package_main_entrypoint() -> None:
    with patch("sys.argv", ["fuzzymatch-finance", "--help"]):
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("fuzzymatch_finance", run_name="__main__")
        assert exc.value.code == 0


def test_column_name_resolution_heuristics() -> None:
    df = pd.DataFrame(
        {
            "custom_rec_id": [1],
            "my_total_amt": [100.50],
            "event_dt": ["2025-01-01"],
            "vendor_memo": ["Test Vendor"],
        }
    )
    res = auto_match_finance(df, df)
    assert len(res.matches) == 1
    assert res.matches.iloc[0]["left_amount"] == 100.50


def test_auto_match_finance_custom_weights() -> None:
    df_l = pd.DataFrame({"id": [1], "name": ["Alpha"], "amount": [10.0]})
    df_r = pd.DataFrame({"id": [2], "name": ["Alpha"], "amount": [10.0]})
    custom_w = {"text": 0.6, "amount": 0.4, "date": 0.0}
    results = auto_match_finance(df_l, df_r, weights=custom_w)
    assert len(results.matches) == 1


def test_auto_match_finance_large_no_text_columns() -> None:
    df_l = pd.DataFrame({"id": range(100), "amount": [10.0] * 100})
    df_r = pd.DataFrame({"id": range(100), "amount": [10.0] * 100})
    results = auto_match_finance(df_l, df_r, budget_cap=500)
    assert isinstance(results, FinanceMatchResults)


def test_auto_match_finance_blocker_empty_fallback() -> None:
    df_l = pd.DataFrame({"id": range(100), "name": [f"Name {i}" for i in range(100)]})
    df_r = pd.DataFrame({"id": range(100), "name": [f"Name {i}" for i in range(100)]})
    from fuzzymatch.blocking.base import CandidateIndex

    empty_ci = CandidateIndex(pairs=[], left_ids=pd.Index([]))
    with patch("fuzzymatch_finance.cli.MultiPassBlock.block", return_value=empty_ci):
        results = auto_match_finance(df_l, df_r)
        assert isinstance(results, FinanceMatchResults)


def test_cli_empty_argv(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "usage:" in captured.err.lower()
