"""Audit Report & Experiment Summary Generator.
Creates Markdown tables and logs satisfying Section 23 Reproducibility Checklist.
"""

from typing import List, Dict, Any
import json
import os
import datetime


class ReportGenerator:
    """Generates auditable Markdown tables and JSON summaries for SIH evaluation."""

    @staticmethod
    def generate_ablation_markdown_table(ablation_results: List[Dict[str, Any]]) -> str:
        """Formats 8-stage ablation results into a clean markdown table."""
        headers = [
            "Configuration", "Candidates", "Inliers", "Inlier Ratio",
            "RMSE (px)", "Grid Cov.", "Distribution", "<0.5px GT %", "Success"
        ]
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in ablation_results:
            success_str = "PASS" if row.get("registration_success") else "FAIL"
            sub05 = f"{row.get('subpixel_pct_under_05px', 0.0):.1f}%"
            row_str = (
                f"| {row.get('configuration', '')} "
                f"| {row.get('n_candidates', 0)} "
                f"| {row.get('n_inliers', 0)} "
                f"| {row.get('inlier_ratio', 0.0):.2f} "
                f"| {row.get('rmse_px', 'N/A')} "
                f"| {row.get('grid_coverage', 0.0):.2f} "
                f"| {row.get('match_distribution_score', 0.0):.2f} "
                f"| {sub05} "
                f"| {success_str} |"
            )
            lines.append(row_str)

        return "\n".join(lines)

    @staticmethod
    def save_experiment_run(
        experiment_id: str,
        pair_id: str,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        output_dir: str = "./experiments/logs"
    ) -> str:
        """Saves experiment run record satisfying Section 23 reproducibility checklist."""
        os.makedirs(output_dir, exist_ok=True)
        record = {
            "experiment_id": experiment_id,
            "pair_id": pair_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metrics": metrics,
            "config": config,
            "status": "COMPLETED"
        }
        out_file = os.path.join(output_dir, f"{experiment_id}_{pair_id}.json")
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)
        return out_file

