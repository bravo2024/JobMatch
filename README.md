# JobMatch

> Talent intelligence platform with job-candidate matching and skill gap analysis.

Generates synthetic job listings and candidate profiles with skills, experience, location, and salary preferences. Trains four classifiers to predict successful job matches, then surfaces ranked recommendations, skill gap analysis, and market intelligence in an interactive dashboard.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.711 |
| Gini | 0.422 |
| KS Statistic | 0.394 |
| F1 Score | 0.554 |
| Accuracy | 0.704 |

5-fold CV AUC: 0.732 ± 0.023. Four models compared.

## Features

| Component | What it does |
|---|---|
| **Candidate Search** | Find matching candidates for any job posting by skills, location, seniority |
| **Job Recommendations** | Top-K job recommendations for any candidate profile |
| **Skill Gap Analysis** | Missing skills identification and learning pathway suggestions |
| **Market Intelligence** | Salary benchmarks, in-demand skills, location trends |

## Repo Structure

```
JobMatch/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic job market dataset: 500 job postings and 1,000 candidate profiles with skills (20 categories), titles, locations, seniority levels, industries, salary ranges, and education levels.

## License

MIT
