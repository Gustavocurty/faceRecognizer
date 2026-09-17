import json
import shutil
from pathlib import Path

import pytest

from recdev.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset_root(tmp_path):
    shutil.copytree(ROOT / "very-easy", tmp_path / "very-easy")
    return tmp_path


def test_evaluate_preserves_separate_runs_and_records_provenance(dataset_root):
    command = ["evaluate", "--root", str(dataset_root), "--dataset", "very-easy", "--method", "pca", "--components", "3"]
    assert main(command) == 0
    assert main(command) == 0
    reports = list((dataset_root / "artifacts" / "runs").rglob("report.json"))
    assert len(reports) == 2
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["manifest_fingerprint"]
    assert len(payload["splits"]) == 2
    assert payload["report"]["config"]["n_components"] == 3
    assert payload["environment"]["versions"]["numpy"]


def test_enroll_identify_persistent_pca_gallery(dataset_root, capsys):
    model = dataset_root / "gallery.npz"
    assert main(["enroll", "--root", str(dataset_root), "--dataset", "very-easy", "--method", "pca",
                 "--components", "3", "--output", str(model)]) == 0
    assert main(["identify", "--root", str(dataset_root), "--model", str(model),
                 "--image", str(dataset_root / "very-easy" / "11-2.jpg")]) == 0
    assert "Identidade: 11" in capsys.readouterr().out


def test_experiment_exports_csv_and_no_automatic_selection(dataset_root):
    assert main(["experiment", "--root", str(dataset_root), "--dataset", "very-easy",
                 "--methods", "pixels", "pca", "--components", "1", "4"]) == 0
    report = next((dataset_root / "artifacts" / "runs").rglob("report.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["experiment"]["selections"] == []
    assert (report.parent / "summary.csv").exists()


def test_rejection_cannot_calibrate_on_easy_test(dataset_root):
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--root", str(dataset_root), "--dataset", "very-easy", "--reject"])
    assert error.value.code == 2
