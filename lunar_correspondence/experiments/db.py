"""Database & Experiment Logging Schema.
Implements the exact schema from Section 22 of the Blueprint using SQLite.
"""

import sqlite3
import os
import json
import datetime
from typing import Dict, Any, List, Optional


class ExperimentDatabase:
    """SQLite database for auditable experiment logging and reproducibility."""

    def __init__(self, db_path: str = "./experiments/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes tables matching Section 22 schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Image
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Image (
                    image_id TEXT PRIMARY KEY,
                    sensor TEXT,
                    file_path TEXT,
                    gsd REAL,
                    footprint_wkt TEXT,
                    sun_az REAL,
                    sun_el REAL,
                    acquisition_time TEXT
                )
            """)

            # ImagePair
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ImagePair (
                    pair_id TEXT PRIMARY KEY,
                    src_image_id TEXT,
                    ref_image_id TEXT,
                    difficulty_bucket TEXT,
                    pair_score REAL,
                    FOREIGN KEY (src_image_id) REFERENCES Image(image_id),
                    FOREIGN KEY (ref_image_id) REFERENCES Image(image_id)
                )
            """)

            # Experiment
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Experiment (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    config_yaml TEXT,
                    git_commit TEXT,
                    matcher_tier_used TEXT,
                    random_seed INTEGER,
                    hardware TEXT
                )
            """)

            # Correspondence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Correspondence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    pair_id TEXT,
                    x REAL,
                    y REAL,
                    x_ref REAL,
                    y_ref REAL,
                    subpixel_dx REAL,
                    subpixel_dy REAL,
                    confidence REAL,
                    residual_error_px REAL,
                    FOREIGN KEY (experiment_id) REFERENCES Experiment(experiment_id),
                    FOREIGN KEY (pair_id) REFERENCES ImagePair(pair_id)
                )
            """)

            # EvaluationResult
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS EvaluationResult (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    pair_id TEXT,
                    inlier_ratio REAL,
                    rmse REAL,
                    median_error REAL,
                    max_error REAL,
                    grid_coverage REAL,
                    confidence_level TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES Experiment(experiment_id),
                    FOREIGN KEY (pair_id) REFERENCES ImagePair(pair_id)
                )
            """)
            conn.commit()

    def record_experiment_run(
        self,
        experiment_id: str,
        pair_id: str,
        matcher_tier_used: str,
        metrics: Dict[str, Any],
        correspondences: List[Dict[str, Any]],
        confidence_level: str = "HIGH",
        hardware: str = "CPU/CUDA",
        random_seed: int = 42
    ):
        """Writes one Experiment row plus EvaluationResult and Correspondences."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert Experiment
            cursor.execute("""
                INSERT OR REPLACE INTO Experiment 
                (experiment_id, timestamp, config_yaml, git_commit, matcher_tier_used, random_seed, hardware)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                datetime.datetime.utcnow().isoformat(),
                "default_config.yaml",
                "HEAD",
                matcher_tier_used,
                random_seed,
                hardware
            ))

            # Insert EvaluationResult
            cursor.execute("""
                INSERT INTO EvaluationResult
                (experiment_id, pair_id, inlier_ratio, rmse, median_error, max_error, grid_coverage, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                pair_id,
                metrics.get("inlier_ratio", 0.0),
                metrics.get("rmse_px", 0.0),
                metrics.get("median_reprojection_error_px", 0.0),
                metrics.get("max_error_px", 0.0),
                metrics.get("grid_coverage", 0.0),
                confidence_level
            ))

            # Insert Correspondences
            for c in correspondences:
                cursor.execute("""
                    INSERT INTO Correspondence
                    (experiment_id, pair_id, x, y, x_ref, y_ref, subpixel_dx, subpixel_dy, confidence, residual_error_px)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    pair_id,
                    c.get("x", 0.0),
                    c.get("y", 0.0),
                    c.get("x_ref", 0.0),
                    c.get("y_ref", 0.0),
                    c.get("subpixel_dx", 0.0),
                    c.get("subpixel_dy", 0.0),
                    c.get("confidence", 1.0),
                    c.get("residual_error_px", 0.0)
                ))

            conn.commit()

