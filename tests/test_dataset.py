import json
from pathlib import Path


def test_dataset_boundary_is_explicit():
    root = Path(__file__).resolve().parents[1]
    records = json.loads(
        (root / "ascend-ops-dataset" / "final" / "test.json").read_text(encoding="utf-8")
    )
    assert len(records) == 682
    executable = [row for row in records if row.get("has_test_scripts")]
    assert len(executable) == 5
    assert {row["name"] for row in executable} == {
        "ascend_add", "ascend_relu", "ascend_sigmoid", "ascend_fmod", "ascend_asinh"
    }
