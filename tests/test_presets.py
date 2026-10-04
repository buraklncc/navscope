import argparse
import sys
import unittest.mock
from pathlib import Path

import pytest

from fund_analyzer.pipeline import main


def test_preset_equity_momentum_25_applies_defaults(monkeypatch, tmp_path):
    test_args = ["pipeline.py", "--preset", "equity-momentum-25", "--data-dir", str(tmp_path)]
    
    # Create fake export file
    export_dir = tmp_path / "tefas_exports"
    export_dir.mkdir()
    (export_dir / "tefas_funds.xlsx").touch()
    
    parsed_args = {}
    
    # Mock run_pipeline to just capture args and return a dummy
    def mock_run_pipeline(*args, **kwargs):
        parsed_args.update(kwargs)
        raise SystemExit(0)  # Stop execution after parsing
        
    monkeypatch.setattr("fund_analyzer.pipeline.run_pipeline", mock_run_pipeline)
    
    with unittest.mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        
    assert parsed_args["provider"] == "tefas-combined"
    assert parsed_args["fund_type"] == "Hisse Senedi Şemsiye Fonu"
    assert parsed_args["risk_min"] == 5
    assert parsed_args["risk_max"] == 7
    assert parsed_args["min_return_1m"] == 0.0
    assert parsed_args["max_funds"] == 25
    assert parsed_args["batch_size"] == 5
    assert parsed_args["sort_by"] == "return_1m_pct"
    assert parsed_args["sort_desc"] is True
    assert parsed_args["export_file"] == str(export_dir / "tefas_funds.xlsx")


def test_preset_explicit_args_override_defaults(monkeypatch, tmp_path):
    test_args = [
        "pipeline.py", 
        "--preset", "equity-momentum-25", 
        "--max-funds", "10", 
        "--fund-type", "Değişken Şemsiye Fonu",
        "--sort-asc",
        "--data-dir", str(tmp_path)
    ]
    
    export_dir = tmp_path / "tefas_exports"
    export_dir.mkdir()
    (export_dir / "tefas_funds.xlsx").touch()
    
    parsed_args = {}
    def mock_run_pipeline(*args, **kwargs):
        parsed_args.update(kwargs)
        raise SystemExit(0)
        
    monkeypatch.setattr("fund_analyzer.pipeline.run_pipeline", mock_run_pipeline)
    
    with unittest.mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        
    assert parsed_args["max_funds"] == 10
    assert parsed_args["fund_type"] == "Değişken Şemsiye Fonu"
    assert parsed_args["sort_desc"] is False
    assert parsed_args["sort_asc"] is True


def test_preset_fails_without_export_file(monkeypatch, tmp_path):
    # Don't create default export file
    test_args = ["pipeline.py", "--preset", "equity-momentum-25", "--data-dir", str(tmp_path)]
    
    with unittest.mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2  # argparse error code

