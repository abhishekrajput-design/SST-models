"""
generate_test_report.py — Generate comprehensive test report visible via API.

Creates a JSON report with all test results:
- Threshold optimization
- API accuracy (30/50/100 calls)
- Desk recording results
- Agent-level breakdown
"""
import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent


def generate_report():
    """Generate comprehensive report from all test results."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "accuracy_metrics": {},
        "threshold_analysis": {},
        "desk_recordings": {},
        "agent_breakdown": {},
    }

    # Load threshold optimization results
    thresh_file = SCRIPT_DIR / "threshold_optimization.json"
    if thresh_file.exists():
        with open(thresh_file) as f:
            thresholds = json.load(f)
        report["threshold_analysis"] = {
            "thresholds_tested": [t["threshold"] for t in thresholds if t.get("calls", 0) > 0],
            "results": thresholds,
        }

    # Load API test results
    api_file = SCRIPT_DIR / "api_test_final.log"
    if api_file.exists():
        with open(api_file) as f:
            content = f.read()
            if "call-level agent ID" in content:
                # Extract accuracy from log
                for line in content.split("\n"):
                    if "call-level agent ID" in line:
                        report["accuracy_metrics"]["api_test"] = line.strip()

    # Load desk recording results
    desk_dir = SCRIPT_DIR.parent / "testing-audio"
    desk_results = []
    if desk_dir.exists():
        for mp3 in desk_dir.rglob("*.mp3"):
            json_file = mp3.with_suffix(".result.json")
            if json_file.exists():
                with open(json_file) as f:
                    result = json.load(f)
                    desk_results.append({
                        "file": mp3.name,
                        "snr_db": result.get("estimated_snr_db"),
                        "bucket": result.get("estimated_bucket"),
                        "agent": result.get("identified_name"),
                        "agent_share": result.get("agent_time_share"),
                    })

    if desk_results:
        report["desk_recordings"] = {
            "count": len(desk_results),
            "results": sorted(desk_results, key=lambda x: x.get("snr_db", 0)),
        }

    # Save report
    report_file = SCRIPT_DIR / "test_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[report] Generated {report_file}")

    # Print summary
    print("\n=== TEST REPORT ===")
    print(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    generate_report()
