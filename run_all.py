import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "code" / "01_main_results.py",
    ROOT / "code" / "02_content_fidelity.py",
    ROOT / "code" / "03_reading_test.py",
    ROOT / "code" / "04_figures.py",
]


def verify_outputs():
    tables = ROOT / "output" / "tables"
    figures = ROOT / "output" / "figures"

    main_res = json.loads((tables / "main_results.json").read_text(encoding="utf-8"))
    assert main_res["run_1"]["cohorts"]["all"]["judges"]["both"]["change"] == 21.1
    assert main_res["run_2"]["cohorts"]["all"]["judges"]["both"]["change"] == 20.8
    assert main_res["run_1"]["cohorts"]["clean2026"]["judges"]["gpt-6.1-sol"]["final"] == 70.8
    assert main_res["run_1"]["cohorts"]["clean2026"]["judges"]["claude-opus-5-5"]["final"] == 55.8
    assert main_res["run_1"]["cohorts"]["all"]["equal_length"]["paragraphs"] == 124
    assert main_res["run_2"]["cohorts"]["all"]["equal_length"]["paragraphs"] == 135
    assert all(main_res["run_1"]["rules"].values())
    assert all(main_res["run_2"]["rules"].values())

    fidelity = json.loads((tables / "content_fidelity.json").read_text(encoding="utf-8"))
    assert fidelity["missing_annotations"] == 0
    assert fidelity["joint"]["gain_joint|gpt-6.1-sol"]["diff"] == 14.1
    assert fidelity["joint"]["gain_joint|claude-opus-5-5"]["diff"] == 19.1
    assert fidelity["decision_rule"]["condition_1_no_fidelity_cost"] is False
    assert fidelity["decision_rule"]["condition_2_joint_gain_above_zero"] is True

    reading = json.loads((tables / "reading_test.json").read_text(encoding="utf-8"))
    assert reading["summary"]["readers"] == 23
    assert reading["summary"]["answers"] == 95
    assert round(100.0 * reading["pooled_diff"]["point"], 1) == 20.8

    for stem in ("fig2_attribution", "figA1_training", "figA2_readers"):
        assert (figures / f"{stem}.pdf").is_file()
        assert (figures / f"{stem}.png").is_file()


def main():
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script)], check=True)
    verify_outputs()
    print("All tables and figures reproduced and verified.")


if __name__ == "__main__":
    main()
