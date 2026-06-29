from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["skill_match_score","experience_years","education_level","role_fit_score","location_preference","salary_expectation_fit","culture_fit_score","commute_distance","industry_experience","job_hopping_freq"]
CATEGORICAL_FEATURES = ["education_level"]
NUMERICAL_FEATURES = ["skill_match_score","experience_years","role_fit_score","location_preference","salary_expectation_fit","culture_fit_score","commute_distance","industry_experience","job_hopping_freq"]
TARGET_NAME = "match_success"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "skill_match_score": rng.beta(5,3,size=n).round(3),
        "experience_years": rng.exponential(scale=5,size=n).clip(0,30).round(1),
        "education_level": rng.choice(["high_school","bachelors","masters","phd","other"],size=n,p=[0.10,0.35,0.35,0.10,0.10]),
        "role_fit_score": rng.beta(5,3,size=n).round(3),
        "location_preference": rng.beta(6,3,size=n).round(3),
        "salary_expectation_fit": rng.beta(5,4,size=n).round(3),
        "culture_fit_score": rng.beta(5,3,size=n).round(3),
        "commute_distance": rng.exponential(scale=15,size=n).clip(0,100).round(1),
        "industry_experience": rng.beta(4,3,size=n).round(3),
        "job_hopping_freq": rng.poisson(lam=1.5,size=n).clip(0,8),
    })
    skill=df["skill_match_score"]; exp=np.clip(df["experience_years"]/30,0,1)
    edu=df["education_level"].map({"high_school":0,"other":0.2,"bachelors":0.4,"masters":0.7,"phd":1}).values
    role=df["role_fit_score"]; loc=df["location_preference"]; sal=df["salary_expectation_fit"]
    cult=df["culture_fit_score"]; comm=np.clip(df["commute_distance"]/100,0,1); ind=df["industry_experience"]
    hop=np.clip(df["job_hopping_freq"]/8,0,1)
    log_odds = -1.0 + 1.0*skill + 0.5*exp + 0.3*edu + 0.8*role + 0.4*loc + 0.5*sal + 0.6*cult - 0.3*comm + 0.3*ind - 0.2*hop + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,70)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(match_success=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
