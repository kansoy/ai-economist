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
    base = main_res["baselines"]
    assert base["D"]["cohorts"]["clean2026"]["judges"]["gpt-6.1-sol"]["rate"] == 6.9
    assert base["D"]["cohorts"]["clean2026"]["judges"]["claude-opus-5-5"]["rate"] == 8.4
    assert base["D"]["cohorts"]["pre2023"]["judges"]["gpt-6.1-sol"]["rate"] == 0.0
    assert base["D"]["cohorts"]["pre2023"]["judges"]["claude-opus-5-5"]["rate"] == 0.0
    assert base["R"]["cohorts"]["clean2026"]["judges"]["gpt-6.1-sol"]["rate"] == 21.2
    assert base["R"]["cohorts"]["clean2026"]["judges"]["claude-opus-5-5"]["rate"] == 7.7
    assert base["R"]["cohorts"]["pre2023"]["judges"]["gpt-6.1-sol"]["rate"] == 1.9
    assert base["R"]["cohorts"]["pre2023"]["judges"]["claude-opus-5-5"]["rate"] == 0.9

    assert main_res["run_1"]["cohorts"]["all"]["judges"]["both"]["change"] == 21.1
    assert main_res["run_2"]["cohorts"]["all"]["judges"]["both"]["change"] == 20.8
    assert main_res["run_1"]["cohorts"]["clean2026"]["judges"]["gpt-6.1-sol"]["final"] == 70.8
    assert main_res["run_1"]["cohorts"]["clean2026"]["judges"]["claude-opus-5-5"]["final"] == 55.8
    assert main_res["run_2"]["cohorts"]["clean2026"]["judges"]["gpt-6.1-sol"]["final"] == 68.2
    assert main_res["run_2"]["cohorts"]["clean2026"]["judges"]["claude-opus-5-5"]["final"] == 54.4
    assert main_res["run_1"]["cohorts"]["pre2023"]["judges"]["gpt-6.1-sol"]["final"] == 31.5
    assert main_res["run_1"]["cohorts"]["pre2023"]["judges"]["claude-opus-5-5"]["final"] == 25.9
    assert main_res["run_2"]["cohorts"]["pre2023"]["judges"]["gpt-6.1-sol"]["final"] == 25.9
    assert main_res["run_2"]["cohorts"]["pre2023"]["judges"]["claude-opus-5-5"]["final"] == 25.0
    assert main_res["run_1"]["cohorts"]["all"]["equal_length"]["paragraphs"] == 124
    assert main_res["run_1"]["cohorts"]["all"]["equal_length"]["change"] == 21.4
    assert main_res["run_2"]["cohorts"]["all"]["equal_length"]["paragraphs"] == 135
    assert main_res["run_2"]["cohorts"]["all"]["equal_length"]["change"] == 19.3
    assert all(main_res["run_1"]["rules"].values())
    assert all(main_res["run_2"]["rules"].values())

    fidelity = json.loads((tables / "content_fidelity.json").read_text(encoding="utf-8"))
    assert fidelity["missing_annotations"] == 0
    assert fidelity["major"]["A|gpt-6.1-sol"]["share"] == 20.9
    assert fidelity["major"]["A|claude-opus-5-5"]["share"] == 0.5
    assert fidelity["major"]["E|gpt-6.1-sol"]["share"] == 8.4
    assert fidelity["major"]["E|claude-opus-5-5"]["share"] == 0.5
    assert fidelity["major"]["B|gpt-6.1-sol"]["share"] == 28.3
    assert fidelity["major"]["B|claude-opus-5-5"]["share"] == 6.8
    assert fidelity["major"]["C|gpt-6.1-sol"]["share"] == 32.5
    assert fidelity["major"]["C|claude-opus-5-5"]["share"] == 9.9
    assert fidelity["major"]["D|gpt-6.1-sol"]["share"] == 28.3
    assert fidelity["major"]["D|claude-opus-5-5"]["share"] == 4.7
    assert fidelity["joint"]["B|gpt-6.1-sol"] == 22.4
    assert fidelity["joint"]["C|gpt-6.1-sol"] == 36.5
    assert fidelity["joint"]["B|claude-opus-5-5"] == 30.6
    assert fidelity["joint"]["C|claude-opus-5-5"] == 49.7
    assert fidelity["joint"]["gain_joint|gpt-6.1-sol"]["diff"] == 14.1
    assert fidelity["joint"]["gain_joint|claude-opus-5-5"]["diff"] == 19.1
    assert fidelity["decision_rule"]["condition_1_no_fidelity_cost"] is False
    assert fidelity["decision_rule"]["condition_2_joint_gain_above_zero"] is True

    reading = json.loads((tables / "reading_test.json").read_text(encoding="utf-8"))
    assert reading["summary"]["readers"] == 23
    assert reading["summary"]["answers"] == 95
    assert round(100.0 * reading["all_draft"]["k"] / reading["all_draft"]["n"], 1) == 61.8
    assert round(100.0 * reading["all_v1"]["k"] / reading["all_v1"]["n"], 1) == 37.9
    assert round(100.0 * reading["all_ral"]["k"] / reading["all_ral"]["n"], 1) == 43.8
    assert round(100.0 * reading["pooled_diff"]["point"], 1) == 20.8

    for name in ("baselines.csv", "table2_main.csv", "tableA2_fidelity.csv", "tableA2_joint.csv"):
        assert (tables / name).stat().st_size > 0
    for stem in ("fig2_attribution", "figA1_training", "figA2_readers"):
        assert (figures / f"{stem}.pdf").stat().st_size > 0
        assert (figures / f"{stem}.png").stat().st_size > 0


def main():
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script)], check=True)
    verify_outputs()
    print("All tables and figures reproduced and verified.")


if __name__ == "__main__":
    main()
