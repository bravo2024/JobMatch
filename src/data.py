"""data.py — Synthetic candidate-job matching data for JobMatch (LinkedIn).

Candidate-job pairs with graded relevance (0-4). Multiple candidates
per job for ranking evaluation. Features include skills, experience,
education, and job requirements.

This is learning-to-rank data, NOT generic tabular classification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def make_synthetic(n_jobs: int = 50, candidates_per_job: int = 20, seed: int = 42) -> dict[str, Any]:
    """Generate candidate-job pairs with graded relevance.

    Relevance is determined by skill overlap, experience match, and
    salary alignment — not a simple binary threshold.
    """
    rng = np.random.default_rng(seed)
    all_skills = ["python", "sql", "ml", "dl", "nlp", "cv", "stats", "spark",
                  "aws", "gcp", "scala", "java", "tableau", "excel", "r"]
    education_levels = ["bachelors", "masters", "phd"]

    rows = []
    for job_id in range(n_jobs):
        # Job requirements
        n_req_skills = rng.integers(3, 6)
        req_skills = set(rng.choice(all_skills, n_req_skills, replace=False))
        req_exp = rng.integers(1, 10)
        req_edu = rng.choice(education_levels)
        req_salary = rng.integers(60, 150) * 1000

        for cand_id in range(candidates_per_job):
            # Candidate profile
            n_cand_skills = rng.integers(2, 8)
            cand_skills = set(rng.choice(all_skills, n_cand_skills, replace=False))
            cand_exp = rng.integers(0, 15)
            cand_edu = rng.choice(education_levels)
            cand_salary = rng.integers(40, 200) * 1000

            # Graded relevance based on match quality
            skill_overlap = len(req_skills & cand_skills) / max(len(req_skills), 1)
            exp_match = 1.0 if cand_exp >= req_exp else cand_exp / req_exp
            edu_match = {"bachelors": 1, "masters": 2, "phd": 3}
            edu_score = min(edu_match[cand_edu], edu_match[req_edu]) / edu_match[req_edu]
            salary_fit = 1.0 - abs(cand_salary - req_salary) / max(req_salary, 1)
            salary_fit = max(0, salary_fit)

            # Relevance grade 0-4
            score = 0.4 * skill_overlap + 0.25 * exp_match + 0.15 * edu_score + 0.20 * salary_fit
            relevance = int(min(4, score * 5))

            rows.append({
                "job_id": job_id, "candidate_id": cand_id,
                "skill_overlap": round(skill_overlap, 3),
                "experience_years": cand_exp, "required_exp": req_exp,
                "education": cand_edu, "required_edu": req_edu,
                "salary": cand_salary, "required_salary": req_salary,
                "relevance": relevance,
            })

    df = pd.DataFrame(rows)
    return {
        "df": df,
        "n_jobs": n_jobs,
        "n_pairs": len(df),
        "candidates_per_job": candidates_per_job,
        "features": ["skill_overlap", "experience_years", "required_exp",
                     "salary", "required_salary"],
        "group_col": "job_id",
        "target_col": "relevance",
    }