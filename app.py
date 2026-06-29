from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobMatch | Talent Intelligence Platform",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded",
)

# ── Dark theme ─────────────────────────────────────────────────────────────────
def _dark():
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor":   "#1e293b",
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  "#e2e8f0",
        "xtick.color":      "#94a3b8",
        "ytick.color":      "#94a3b8",
        "text.color":       "#e2e8f0",
        "grid.color":       "#334155",
        "legend.facecolor": "#1e293b",
        "legend.edgecolor": "#334155",
    })
_dark()

PALETTE = ["#38bdf8","#818cf8","#34d399","#fb923c","#f472b6","#facc15","#a78bfa","#22d3ee"]

ALL_SKILLS  = ["Python","SQL","ML","TensorFlow","AWS","Spark","NLP","Statistics",
               "Leadership","Product","Docker","React","PyTorch","Scala","GCP",
               "Kubernetes","Java","R","Tableau","Excel"]
JOB_TITLES  = ["Data Scientist","ML Engineer","Software Engineer","Data Analyst","Product Manager"]
LOCATIONS   = ["San Francisco","New York","Austin","Seattle","Remote"]
SENIORITY   = ["junior","mid","senior","staff"]
INDUSTRIES  = ["Tech","Finance","Healthcare","Retail","Media"]
EDUCATION   = ["BS","MS","PhD"]
COMPANY_SIZES = ["startup","mid","enterprise"]
SKILL_GROUPS = {
    "Programming": ["Python","Java","R","Scala","React"],
    "Cloud":       ["AWS","GCP","Docker","Kubernetes"],
    "Deep Learning":["TensorFlow","PyTorch","ML"],
    "Data":        ["SQL","Spark","Tableau","Excel"],
    "Soft":        ["Leadership","Product"],
    "Science":     ["Statistics","NLP"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Generating synthetic data…")
def generate_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    N_JOBS, N_CANDS = 500, 1000

    jobs = []
    for i in range(N_JOBS):
        title    = rng.choice(JOB_TITLES)
        n_sk     = int(rng.integers(5, 9))
        skills   = list(rng.choice(ALL_SKILLS, size=n_sk, replace=False))
        sal_min  = int(rng.integers(60, 140)) * 1000
        sal_max  = sal_min + int(rng.integers(20, 60)) * 1000
        exp_req  = int(rng.integers(0, 12))
        jobs.append(dict(job_id=i, title=title,
                         company_size=rng.choice(COMPANY_SIZES),
                         location=rng.choice(LOCATIONS),
                         seniority=rng.choice(SENIORITY),
                         required_skills=skills,
                         salary_min=sal_min, salary_max=sal_max,
                         industry=rng.choice(INDUSTRIES),
                         exp_required=exp_req))
    jobs_df = pd.DataFrame(jobs)

    cands = []
    for i in range(N_CANDS):
        n_sk   = int(rng.integers(5, 11))
        skills = list(rng.choice(ALL_SKILLS, size=n_sk, replace=False))
        yoe    = float(rng.uniform(0, 20))
        c_sal  = int(rng.integers(50, 180)) * 1000
        cands.append(dict(cand_id=i,
                          current_role=rng.choice(JOB_TITLES),
                          years_experience=round(yoe, 1),
                          skills=skills,
                          education=rng.choice(EDUCATION),
                          preferred_locations=list(rng.choice(LOCATIONS,
                              size=int(rng.integers(1, 4)), replace=False)),
                          preferred_seniority=rng.choice(SENIORITY),
                          current_salary=c_sal))
    cands_df = pd.DataFrame(cands)

    # 5 000 pair labels
    n_pairs = 5000
    job_ids  = rng.integers(0, N_JOBS,  size=n_pairs)
    cand_ids = rng.integers(0, N_CANDS, size=n_pairs)
    pairs = []
    for ji, ci in zip(job_ids, cand_ids):
        job  = jobs[ji];  cand = cands[ci]
        jsk  = set(job["required_skills"]); csk = set(cand["skills"])
        ovlp = len(jsk & csk) / len(jsk) if jsk else 0
        edel = abs(cand["years_experience"] - job["exp_required"])
        loc  = int(job["location"] in cand["preferred_locations"]
                   or "Remote" in cand["preferred_locations"])
        edu  = 1 if cand["education"] in ("MS","PhD") else 0
        sen  = 1 if cand["preferred_seniority"] == job["seniority"] else 0
        cs   = cand["current_salary"]
        sal  = 1.0 if job["salary_min"] <= cs <= job["salary_max"] else max(
            0, 1 - min(abs(cs-job["salary_min"]), abs(cs-job["salary_max"])) / 50000)
        lbl  = int(ovlp >= 0.6 and edel <= 2 and bool(loc))
        pairs.append(dict(job_id=ji, cand_id=ci,
                          skill_overlap_pct=round(ovlp*100,2),
                          experience_delta=round(edel,2),
                          location_match=loc,
                          education_match=edu,
                          seniority_match=sen,
                          salary_alignment=round(sal,3),
                          label=lbl))
    pairs_df = pd.DataFrame(pairs)
    return jobs_df, cands_df, pairs_df

# ══════════════════════════════════════════════════════════════════════════════
#  TF-IDF FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Building TF-IDF index…")
def build_tfidf(_jobs_df):
    docs  = [" ".join(r) for r in _jobs_df["required_skills"]]
    vocab = sorted({w for d in docs for w in d.split()})
    V, N  = len(vocab), len(docs)
    w2i   = {w: i for i, w in enumerate(vocab)}

    tf = np.zeros((N, V))
    for di, doc in enumerate(docs):
        words = doc.split()
        for w in words:
            tf[di, w2i[w]] += 1
        tf[di] /= max(len(words), 1)

    df_c = (tf > 0).sum(axis=0)
    idf  = np.log(N / (1 + df_c)) + 1
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-10
    return tfidf / norms, vocab, w2i, idf

# ══════════════════════════════════════════════════════════════════════════════
#  LOGISTIC REGRESSION (SGD)  +  RANKNET
# ══════════════════════════════════════════════════════════════════════════════
FEAT_COLS = ["skill_overlap_pct","experience_delta","location_match",
             "education_match","seniority_match","salary_alignment"]

@st.cache_resource(show_spinner="Training ranking model…")
def train_model(_pairs_df, skill_w: float = 1.0, exp_w: float = 1.0):
    df  = _pairs_df.dropna().copy()
    X   = df[FEAT_COLS].values.astype(float)
    y   = df["label"].values.astype(float)
    mu  = X.mean(0); sig = X.std(0) + 1e-8
    Xn  = (X - mu) / sig
    Xn[:, 0] *= skill_w
    Xn[:, 1] *= exp_w

    n   = len(Xn)
    idx = np.random.default_rng(42).permutation(n)
    sp  = int(0.8 * n)
    tr, te = idx[:sp], idx[sp:]
    Xtr, ytr = Xn[tr], y[tr]
    Xte, yte = Xn[te], y[te]

    # Pointwise SGD logistic
    W = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(100):
        z   = Xtr @ W + b
        p   = 1 / (1 + np.exp(-np.clip(z, -20, 20)))
        err = p - ytr
        W  -= 0.05 * (Xtr.T @ err) / len(ytr)
        b  -= 0.05 * err.mean()

    # Pairwise RankNet
    rn_losses = []
    Wrn = W.copy(); brn = b
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    np2 = min(500, len(pos), len(neg))
    rng2 = np.random.default_rng(0)
    for _ in range(50):
        pi = rng2.choice(pos, np2, replace=True)
        ni = rng2.choice(neg, np2, replace=True)
        si = Xtr[pi] @ Wrn + brn
        sj = Xtr[ni] @ Wrn + brn
        sg = 1 / (1 + np.exp(-np.clip(si - sj, -20, 20)))
        rn_losses.append(-np.log(sg + 1e-10).mean())
        gr = sg - 1
        Wrn -= 0.01 * (Xtr[pi] - Xtr[ni]).T @ gr / np2
        brn -= 0.01 * gr.mean()

    scores_te = 1 / (1 + np.exp(-(Xte @ W + b)))
    return dict(W=W, b=b, mu=mu, sig=sig,
                scores_te=scores_te, y_te=yte,
                ranknet_losses=rn_losses)

# ══════════════════════════════════════════════════════════════════════════════
#  RANKING METRICS (manual)
# ══════════════════════════════════════════════════════════════════════════════
def dcg_at_k(rel, k):
    r = np.asarray(rel[:k], dtype=float)
    return float(np.sum(r / np.log2(np.arange(2, len(r)+2))))

def ndcg_at_k(scores, labels, k):
    o    = np.argsort(-scores)
    idl  = np.sort(labels)[::-1]
    dcg  = dcg_at_k(labels[o], k)
    idcg = dcg_at_k(idl, k)
    return dcg / idcg if idcg > 0 else 0.0

def avg_precision(scores, labels, k):
    o = np.argsort(-scores)[:k]
    hits = 0; total = 0.0
    for i, idx in enumerate(o):
        if labels[idx] == 1:
            hits += 1; total += hits / (i+1)
    return total / max(1, labels.sum())

def compute_metrics(scores, labels, Ks=(1,3,5,10)):
    chunk  = 20
    groups = [(scores[i:i+chunk], labels[i:i+chunk])
              for i in range(0, len(scores), chunk)
              if len(scores[i:i+chunk]) >= 2]
    valid  = [(s, l) for s, l in groups if l.sum() > 0]
    if not valid:
        return {}
    vs, vl = zip(*valid)
    out = {}
    for k in Ks:
        out[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(s,l,k) for s,l in zip(vs,vl)]))
        out[f"MAP@{k}"]  = float(np.mean([avg_precision(s,l,k) for s,l in zip(vs,vl)]))
    rr = []
    for s, l in zip(vs, vl):
        for rank, idx in enumerate(np.argsort(-s)):
            if l[idx] == 1:
                rr.append(1/(rank+1)); break
        else:
            rr.append(0.0)
    out["MRR"] = float(np.mean(rr))
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _features(job_row, cand_row):
    jsk  = set(job_row["required_skills"]); csk = set(cand_row["skills"])
    ovlp = len(jsk & csk) / len(jsk) if jsk else 0
    edel = abs(cand_row["years_experience"] - job_row["exp_required"])
    loc  = int(job_row["location"] in cand_row["preferred_locations"]
               or "Remote" in cand_row["preferred_locations"])
    edu  = 1 if cand_row["education"] in ("MS","PhD") else 0
    sen  = 1 if cand_row["preferred_seniority"] == job_row["seniority"] else 0
    cs   = cand_row["current_salary"]
    sal  = 1.0 if job_row["salary_min"] <= cs <= job_row["salary_max"] else max(
        0, 1-min(abs(cs-job_row["salary_min"]), abs(cs-job_row["salary_max"]))/50000)
    return np.array([ovlp*100, edel, loc, edu, sen, sal], dtype=float), jsk - csk

def _score(x_raw, W, b, mu, sig, sw=1.0, ew=1.0):
    xn = (x_raw - mu) / sig
    xn[0] *= sw; xn[1] *= ew
    return float(1 / (1 + np.exp(-(xn @ W + b))))

def score_candidates(job_row, cands_df, W, b, mu, sig, sw=1.0, ew=1.0):
    rows = []
    for _, c in cands_df.iterrows():
        x, miss = _features(job_row, c)
        sc = _score(x, W, b, mu, sig, sw, ew)
        rows.append(dict(cand_id=int(c.cand_id), current_role=c.current_role,
                         education=c.education, years_exp=c.years_experience,
                         skill_overlap_pct=round(x[0],1), exp_delta=round(x[1],1),
                         location_match=bool(x[2]), salary_alignment=round(x[5],2),
                         score=round(sc,4), missing_skills=list(miss)))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

def score_jobs(cand_row, jobs_df, W, b, mu, sig, sw=1.0, ew=1.0):
    rows = []
    for _, j in jobs_df.iterrows():
        x, _ = _features(j, cand_row)
        sc   = _score(x, W, b, mu, sig, sw, ew)
        rows.append(dict(job_id=int(j.job_id), title=j.title,
                         location=j.location, seniority=j.seniority,
                         salary_min=j.salary_min, salary_max=j.salary_max,
                         skill_overlap_pct=round(x[0],1), exp_delta=round(x[1],1),
                         location_match=bool(x[2]), score=round(sc,4)))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

def mmr_rerank(df, lam=0.7, top_n=10):
    pool = df.head(30).reset_index(drop=True)
    cols = [c for c in ["skill_overlap_pct","exp_delta","location_match","salary_alignment"]
            if c in pool.columns]
    M    = pool[cols].values.astype(float)
    M    = (M - M.mean(0)) / (M.std(0) + 1e-8)
    sel, rem = [], list(range(len(pool)))
    for _ in range(min(top_n, len(pool))):
        if not rem: break
        if not sel:
            best = max(rem, key=lambda i: pool.loc[i,"score"])
        else:
            def mmr_s(i):
                sv = M[sel]; v = M[i]
                nv = np.linalg.norm(v)
                sims = sv @ v / (np.linalg.norm(sv,axis=1)*nv + 1e-10)
                return lam*pool.loc[i,"score"] - (1-lam)*sims.max()
            best = max(rem, key=mmr_s)
        sel.append(best); rem.remove(best)
    return pool.iloc[sel].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Controls")
    K_slider  = st.slider("Ranking K", 1, 10, 5)
    skill_w   = st.slider("Skill Weight",      0.1, 3.0, 1.0, 0.1)
    exp_w     = st.slider("Experience Weight",  0.1, 3.0, 1.0, 0.1)
    min_match = st.slider("Min Match Score",    0.0, 1.0, 0.0, 0.05)
    mmr_lam   = st.slider("MMR λ (diversity)",  0.0, 1.0, 0.7, 0.05)
    st.markdown("---")
    st.caption("JobMatch · LinkedIn/Indeed-style NLP Ranking")

# ── Load data & models ─────────────────────────────────────────────────────────
jobs_df, cands_df, pairs_df = generate_data()
tfidf_mat, vocab, w2i, idf  = build_tfidf(jobs_df)
model    = train_model(pairs_df, skill_w, exp_w)
W        = model["W"]; B = model["b"]
mu       = model["mu"]; sig = model["sig"]
metrics  = compute_metrics(model["scores_te"], model["y_te"])

with st.sidebar:
    sel_job_idx  = st.selectbox("Select Job (Tab 4)",  range(len(jobs_df)),
        format_func=lambda i: f"#{i} {jobs_df.loc[i,'title']} ({jobs_df.loc[i,'location']})")
    sel_cand_idx = st.selectbox("Select Candidate (Tab 4)", range(len(cands_df)),
        format_func=lambda i: f"#{i} {cands_df.loc[i,'current_role']} ({cands_df.loc[i,'years_experience']}yr)")

# ── Header KPIs ────────────────────────────────────────────────────────────────
st.title("🎯 JobMatch · AI Talent Intelligence Platform")
st.caption("LinkedIn/Indeed-style NLP-powered Job-Candidate Ranking · TF-IDF + Learning-to-Rank")

match_rate  = pairs_df["label"].mean()
ndcg5       = metrics.get("NDCG@5", 0.0)
mrr_val     = metrics.get("MRR",    0.0)
avg_overlap = pairs_df["skill_overlap_pct"].mean()

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("📋 Jobs",            f"{len(jobs_df):,}")
k2.metric("👤 Candidates",       f"{len(cands_df):,}")
k3.metric("✅ Match Rate",       f"{match_rate:.1%}")
k4.metric("📈 NDCG@5",          f"{ndcg5:.4f}")
k5.metric("🔁 MRR",             f"{mrr_val:.4f}")
k6.metric("🔗 Avg Skill Overlap",f"{avg_overlap:.1f}%")
st.markdown("---")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📄 Data Explorer — Jobs & Candidates",
    "🔤 NLP & TF-IDF Skill Matching",
    "🤖 Learning-to-Rank Model",
    "🎯 Candidate-Job Recommendations",
    "📊 Market Intelligence",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📄 Synthetic Dataset Overview")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Sample Job Listings (first 10)**")
        dj = jobs_df.head(10).copy()
        dj["required_skills"] = dj["required_skills"].apply(", ".join)
        st.dataframe(dj[["title","company_size","location","seniority",
                          "required_skills","salary_min","salary_max","industry"]],
                     use_container_width=True, height=220)
    with c2:
        st.markdown("**Sample Candidate Profiles (first 10)**")
        dc = cands_df.head(10).copy()
        dc["skills"]             = dc["skills"].apply(", ".join)
        dc["preferred_locations"]= dc["preferred_locations"].apply(", ".join)
        st.dataframe(dc[["current_role","years_experience","skills",
                          "education","preferred_seniority","current_salary"]],
                     use_container_width=True, height=220)

    st.markdown("---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0f172a")

    # Job title distribution
    ax = axes[0,0]; ax.set_facecolor("#1e293b")
    tc  = jobs_df["title"].value_counts()
    brs = ax.bar(tc.index, tc.values, color=PALETTE[:len(tc)])
    ax.set_title("Job Title Distribution", color="white", fontsize=13)
    ax.tick_params(axis="x", rotation=20, colors="#94a3b8")
    ax.tick_params(axis="y", colors="#94a3b8")
    for br, v in zip(brs, tc.values):
        ax.text(br.get_x()+br.get_width()/2, v+1, str(v), ha="center", color="white", fontsize=9)

    # Top-20 skill frequency
    ax = axes[0,1]; ax.set_facecolor("#1e293b")
    sk_cnt  = Counter(s for row in jobs_df["required_skills"] for s in row)
    top20   = sk_cnt.most_common(20)
    sk_n, sk_v = zip(*top20)
    c20 = plt.cm.plasma(np.linspace(0.2, 0.9, 20))
    ax.barh(sk_n, sk_v, color=c20)
    ax.set_title("Top 20 Skills in Job Listings", color="white", fontsize=13)
    ax.tick_params(colors="#94a3b8"); ax.invert_yaxis()

    # Candidate experience histogram
    ax = axes[1,0]; ax.set_facecolor("#1e293b")
    ax.hist(cands_df["years_experience"], bins=20, color="#38bdf8",
            edgecolor="#0f172a", alpha=0.85)
    ax.set_title("Candidate Years of Experience", color="white", fontsize=13)
    ax.set_xlabel("Years", color="#94a3b8")
    ax.set_ylabel("Count",  color="#94a3b8")
    ax.tick_params(colors="#94a3b8")

    # 10×10 skill-overlap heatmap
    ax = axes[1,1]; ax.set_facecolor("#1e293b")
    sj = jobs_df.sample(10, random_state=1).reset_index(drop=True)
    sc = cands_df.sample(10, random_state=2).reset_index(drop=True)
    heat = np.zeros((10,10))
    for ri, jr in sj.iterrows():
        for ci, cr in sc.iterrows():
            jsk = set(jr["required_skills"]); csk = set(cr["skills"])
            heat[ri,ci] = len(jsk&csk)/len(jsk) if jsk else 0
    im = ax.imshow(heat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_title("10×10 Skill Overlap Heatmap\n(Sample Jobs × Candidates)",
                 color="white", fontsize=12)
    ax.set_xlabel("Candidates", color="#94a3b8")
    ax.set_ylabel("Jobs",       color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)
    st.metric("Overall Match Rate (skill≥60% + exp±2yr + location)", f"{match_rate:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — NLP & TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔤 TF-IDF Skill Matching — From Scratch (NumPy)")
    st.latex(r"\text{TF}(t,d)=\frac{\text{count}(t,d)}{|d|}")
    st.latex(r"\text{IDF}(t)=\log\!\left(\frac{N}{1+\text{df}(t)}\right)+1")
    st.latex(r"\text{TF-IDF}(t,d)=\text{TF}(t,d)\times\text{IDF}(t)")
    st.latex(r"\text{CosSim}(\mathbf{j},\mathbf{c})=\frac{\mathbf{j}\cdot\mathbf{c}}{\|\mathbf{j}\|\;\|\mathbf{c}\|}")
    st.markdown("---")

    job_sel = st.selectbox("Select Job for TF-IDF Analysis", range(len(jobs_df)),
        format_func=lambda i: f"#{i} — {jobs_df.loc[i,'title']} ({jobs_df.loc[i,'location']})",
        key="tfidf_sel")
    sel_job = jobs_df.loc[job_sel]

    # Build candidate TF-IDF vectors on demand
    @st.cache_resource(show_spinner=False)
    def cand_tfidf(_cands_df, _idf, _w2i, _vocab):
        vecs = []
        for _, cr in _cands_df.iterrows():
            doc   = " ".join(cr["skills"]); words = doc.split()
            v     = np.zeros(len(_vocab))
            for w in words:
                if w in _w2i: v[_w2i[w]] += 1
            v /= max(len(words),1); v = v * _idf
            nrm = np.linalg.norm(v)
            vecs.append(v/nrm if nrm > 1e-10 else v)
        return np.array(vecs)
    cand_vecs = cand_tfidf(cands_df, idf, w2i, vocab)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**TF-IDF Top Terms — Job #{job_sel}: {sel_job['title']}**")
        jvec   = tfidf_mat[job_sel]
        top_i  = np.argsort(-jvec)[:15]
        t_names= [vocab[i] for i in top_i]
        t_vals = jvec[top_i]
        fig_a, ax_a = plt.subplots(figsize=(6,4))
        fig_a.patch.set_facecolor("#0f172a"); ax_a.set_facecolor("#1e293b")
        ax_a.barh(t_names[::-1], t_vals[::-1],
                  color=plt.cm.cool(np.linspace(0.3,0.9,len(t_names))))
        ax_a.set_title(f"TF-IDF Weights — {sel_job['title']}", color="white", fontsize=12)
        ax_a.tick_params(colors="#94a3b8")
        ax_a.set_xlabel("TF-IDF score", color="#94a3b8")
        st.pyplot(fig_a)

    with col_b:
        sims     = cand_vecs @ tfidf_mat[job_sel]
        best_ci  = int(np.argmax(sims))
        best_c   = cands_df.loc[best_ci]
        missing  = list(set(sel_job["required_skills"]) - set(best_c["skills"]))
        st.markdown(f"**Skill Gap — Best Candidate #{best_ci}**")
        st.write(f"**Role:** {best_c['current_role']} | **YoE:** {best_c['years_experience']} | **Edu:** {best_c['education']}")
        st.write(f"**Candidate Skills:** {', '.join(best_c['skills'])}")
        st.write(f"**Job Required:**     {', '.join(sel_job['required_skills'])}")
        if missing:
            st.warning(f"Missing Skills: {', '.join(missing)}")
        else:
            st.success("Full skill match!")

        top10s = np.sort(sims)[::-1][:10]
        fig_b, ax_b = plt.subplots(figsize=(6,3))
        fig_b.patch.set_facecolor("#0f172a"); ax_b.set_facecolor("#1e293b")
        ax_b.plot(range(1,11), top10s, "o-", color="#38bdf8", linewidth=2)
        ax_b.fill_between(range(1,11), top10s, alpha=0.3, color="#38bdf8")
        ax_b.set_title("TF-IDF Cosine Sim — Top 10 Candidates", color="white", fontsize=11)
        ax_b.set_xlabel("Rank", color="#94a3b8"); ax_b.set_ylabel("Cosine Sim", color="#94a3b8")
        ax_b.tick_params(colors="#94a3b8")
        st.pyplot(fig_b)

    st.markdown("---")
    st.subheader("Skill Taxonomy & Demand vs Supply")
    col_c, col_d = st.columns(2)

    with col_c:
        gdf = pd.DataFrame({
            grp: {
                "Jobs":       sum(1 for r in jobs_df["required_skills"]  for s in r if s in sk),
                "Candidates": sum(1 for r in cands_df["skills"]          for s in r if s in sk),
            }
            for grp, sk in SKILL_GROUPS.items()
        }).T
        fig_c, ax_c = plt.subplots(figsize=(6,4))
        fig_c.patch.set_facecolor("#0f172a"); ax_c.set_facecolor("#1e293b")
        xc = np.arange(len(gdf))
        ax_c.bar(xc-0.2, gdf["Jobs"],       0.35, label="Jobs",       color="#38bdf8", alpha=0.85)
        ax_c.bar(xc+0.2, gdf["Candidates"], 0.35, label="Candidates", color="#34d399", alpha=0.85)
        ax_c.set_xticks(xc); ax_c.set_xticklabels(gdf.index, rotation=20, color="#94a3b8")
        ax_c.tick_params(axis="y", colors="#94a3b8")
        ax_c.set_title("Skill Group: Demand vs Supply", color="white", fontsize=12)
        ax_c.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        st.pyplot(fig_c)

    with col_d:
        all_freq = sorted(sk_cnt.values(), reverse=True)
        ranks    = np.arange(1, len(all_freq)+1)
        st.latex(r"f(r)\propto r^{-\alpha}")
        log_r = np.log(ranks); log_f = np.log(np.array(all_freq, dtype=float)+1e-10)
        coeffs = np.polyfit(log_r, log_f, 1)
        alpha_fit = -coeffs[0]
        fit_line  = np.exp(coeffs[1]) * ranks**(-alpha_fit)
        fig_d, ax_d = plt.subplots(figsize=(6,4))
        fig_d.patch.set_facecolor("#0f172a"); ax_d.set_facecolor("#1e293b")
        ax_d.loglog(ranks, all_freq, "o", color="#fb923c", alpha=0.7, markersize=6)
        ax_d.loglog(ranks, fit_line, "--", color="#f472b6", linewidth=2,
                    label=f"α = {alpha_fit:.2f}")
        ax_d.set_title("Zipf's Law — Skill Frequency", color="white", fontsize=12)
        ax_d.set_xlabel("Rank", color="#94a3b8"); ax_d.set_ylabel("Frequency", color="#94a3b8")
        ax_d.tick_params(colors="#94a3b8")
        ax_d.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
        st.pyplot(fig_d)

    st.subheader("Individual Skill Demand vs Supply (Top 15)")
    top15_names = [s for s,_ in sk_cnt.most_common(15)]
    jd_vals = [sk_cnt[s] for s in top15_names]
    cs_vals = [sum(1 for r in cands_df["skills"] for sk2 in r if sk2==s) for s in top15_names]
    fig_e, ax_e = plt.subplots(figsize=(13,4))
    fig_e.patch.set_facecolor("#0f172a"); ax_e.set_facecolor("#1e293b")
    xe = np.arange(len(top15_names))
    ax_e.bar(xe-0.2, jd_vals, 0.35, label="Job Demand",       color="#818cf8", alpha=0.9)
    ax_e.bar(xe+0.2, cs_vals, 0.35, label="Candidate Supply", color="#34d399", alpha=0.9)
    ax_e.set_xticks(xe); ax_e.set_xticklabels(top15_names, rotation=30, color="#94a3b8", ha="right")
    ax_e.tick_params(axis="y", colors="#94a3b8")
    ax_e.set_title("Top 15 Skills: Demand vs Supply", color="white", fontsize=13)
    ax_e.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
    st.pyplot(fig_e)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LEARNING-TO-RANK
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Learning-to-Rank — Logistic SGD + RankNet (NumPy only)")
    st.latex(r"P(\text{match})=\sigma\!\left(\mathbf{w}^\top\mathbf{x}+b\right)=\frac{1}{1+e^{-(\mathbf{w}^\top\mathbf{x}+b)}}")
    st.latex(r"\mathcal{L}_{\text{pointwise}}=-\frac{1}{N}\sum_{i}\left[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\right]")
    st.latex(r"\mathcal{L}_{\text{RankNet}}=-\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\log\sigma(s_i-s_j)")
    st.latex(r"\text{DCG}@K=\sum_{i=1}^{K}\frac{\text{rel}_i}{\log_2(i+1)},\quad\text{NDCG}@K=\frac{\text{DCG}@K}{\text{IDCG}@K}")
    st.latex(r"\text{MRR}=\frac{1}{|Q|}\sum_{q}\frac{1}{\text{rank}_q}")
    st.markdown("---")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("**Ranking Metrics at K = 1, 3, 5, 10**")
        met_rows = []
        for k in [1,3,5,10]:
            met_rows.append({"K": k,
                             "NDCG@K": round(metrics.get(f"NDCG@{k}",0),4),
                             "MAP@K":  round(metrics.get(f"MAP@{k}",0),4),
                             "MRR":    round(metrics.get("MRR",0),4) if k==5 else "—"})
        st.dataframe(pd.DataFrame(met_rows).set_index("K"), use_container_width=True)

        ks = [1,3,5,10]
        ndcg_v = [metrics.get(f"NDCG@{k}",0) for k in ks]
        map_v  = [metrics.get(f"MAP@{k}",0)  for k in ks]
        fig_l1, ax_l1 = plt.subplots(figsize=(6,3.5))
        fig_l1.patch.set_facecolor("#0f172a"); ax_l1.set_facecolor("#1e293b")
        ax_l1.plot(ks, ndcg_v, "o-", color="#38bdf8", linewidth=2, label="NDCG@K")
        ax_l1.plot(ks, map_v,  "s--",color="#34d399", linewidth=2, label="MAP@K")
        ax_l1.axhline(metrics.get("MRR",0), color="#fb923c", linestyle=":",
                      linewidth=1.5, label=f"MRR={metrics.get('MRR',0):.3f}")
        ax_l1.set_xticks(ks); ax_l1.tick_params(colors="#94a3b8")
        ax_l1.set_xlabel("K",color="#94a3b8"); ax_l1.set_ylabel("Score",color="#94a3b8")
        ax_l1.set_title("Ranking Metrics vs K", color="white", fontsize=12)
        ax_l1.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
        ax_l1.set_ylim(0,1)
        st.pyplot(fig_l1)

    with col_l2:
        st.markdown("**Feature Importance (LR Coefficients)**")
        feat_labels = ["Skill Overlap%","Exp Delta","Location","Education","Seniority","Salary Align"]
        order = np.argsort(W)
        fig_l2, ax_l2 = plt.subplots(figsize=(6,4))
        fig_l2.patch.set_facecolor("#0f172a"); ax_l2.set_facecolor("#1e293b")
        colors_l2 = ["#34d399" if v>=0 else "#f43f5e" for v in W[order]]
        ax_l2.barh([feat_labels[i] for i in order], W[order], color=colors_l2)
        ax_l2.axvline(0, color="#94a3b8", linewidth=1)
        ax_l2.set_title("Feature Coefficients", color="white", fontsize=12)
        ax_l2.tick_params(colors="#94a3b8")
        ax_l2.set_xlabel("Coefficient", color="#94a3b8")
        st.pyplot(fig_l2)

        st.markdown("**RankNet Pairwise Loss Curve**")
        fig_l3, ax_l3 = plt.subplots(figsize=(6,2.8))
        fig_l3.patch.set_facecolor("#0f172a"); ax_l3.set_facecolor("#1e293b")
        rn_loss = model["ranknet_losses"]
        ax_l3.plot(rn_loss, color="#f472b6", linewidth=2)
        ax_l3.fill_between(range(len(rn_loss)), rn_loss, alpha=0.2, color="#f472b6")
        ax_l3.set_title("RankNet Loss per Epoch", color="white", fontsize=12)
        ax_l3.set_xlabel("Epoch",color="#94a3b8")
        ax_l3.set_ylabel("−log σ(sᵢ−sⱼ)",color="#94a3b8")
        ax_l3.tick_params(colors="#94a3b8")
        st.pyplot(fig_l3)

    st.markdown("---"); st.subheader("Predicted Score Distribution — Test Set")
    sc_te = model["scores_te"]; y_te = model["y_te"]
    pos_sc = sc_te[y_te==1]; neg_sc = sc_te[y_te==0]
    fig_l4, ax_l4 = plt.subplots(figsize=(10,3.5))
    fig_l4.patch.set_facecolor("#0f172a"); ax_l4.set_facecolor("#1e293b")
    ax_l4.hist(neg_sc, bins=40, alpha=0.6, color="#f43f5e", label="No Match", density=True)
    ax_l4.hist(pos_sc, bins=40, alpha=0.6, color="#34d399", label="Match",    density=True)
    ax_l4.set_title("Score Distribution (Test Set)", color="white", fontsize=13)
    ax_l4.set_xlabel("P(match)",color="#94a3b8"); ax_l4.set_ylabel("Density",color="#94a3b8")
    ax_l4.tick_params(colors="#94a3b8")
    ax_l4.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
    st.pyplot(fig_l4)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎯 Candidate-Job Recommendations")
    sel_job_row  = jobs_df.loc[sel_job_idx]
    sel_cand_row = cands_df.loc[sel_cand_idx]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"### Top Candidates for Job #{sel_job_idx}: **{sel_job_row['title']}**")
        st.write(f"📍 {sel_job_row['location']} | 🏢 {sel_job_row['company_size']} | "
                 f"💼 {sel_job_row['seniority']} | "
                 f"💰 ${sel_job_row['salary_min']//1000}k–${sel_job_row['salary_max']//1000}k")
        st.write(f"**Required Skills:** {', '.join(sel_job_row['required_skills'])}")
        ranked_c = score_candidates(sel_job_row, cands_df, W, B, mu, sig, skill_w, exp_w)
        ranked_c = ranked_c[ranked_c["score"] >= min_match].head(10)
        st.dataframe(ranked_c[["cand_id","current_role","education","years_exp",
                                "skill_overlap_pct","exp_delta","location_match",
                                "salary_alignment","score"]],
                     use_container_width=True, height=280)
        if len(ranked_c):
            top_c = ranked_c.iloc[0]
            miss  = top_c["missing_skills"]
            if miss:
                st.warning(f"Top candidate should improve: **{', '.join(miss)}**")
            else:
                st.success("Top candidate has all required skills!")

    with col_r2:
        st.markdown(f"### Top Jobs for Candidate #{sel_cand_idx}: **{sel_cand_row['current_role']}**")
        st.write(f"🎓 {sel_cand_row['education']} | 📅 {sel_cand_row['years_experience']} yrs | "
                 f"💰 ${sel_cand_row['current_salary']//1000}k")
        st.write(f"**Skills:** {', '.join(sel_cand_row['skills'])}")
        st.write(f"**Preferred:** {', '.join(sel_cand_row['preferred_locations'])}")
        ranked_j = score_jobs(sel_cand_row, jobs_df, W, B, mu, sig, skill_w, exp_w)
        ranked_j = ranked_j[ranked_j["score"] >= min_match].head(10)
        st.dataframe(ranked_j[["job_id","title","location","seniority",
                                "salary_min","salary_max","skill_overlap_pct",
                                "exp_delta","location_match","score"]],
                     use_container_width=True, height=280)

    st.markdown("---")
    col_r3, col_r4 = st.columns(2)
    with col_r3:
        st.subheader("Feature Contribution Waterfall")
        if len(ranked_c):
            top_row = ranked_c.iloc[0]
            cr      = cands_df.loc[int(top_row["cand_id"])]
            x_raw, _ = _features(sel_job_row, cr)
            xn       = (x_raw - mu) / sig
            xn[0]   *= skill_w; xn[1] *= exp_w
            contribs = xn * W
            feat_labels = ["Skill%","ExpDelta","Location","Education","Seniority","Salary"]
            fig_r1, ax_r1 = plt.subplots(figsize=(6,4))
            fig_r1.patch.set_facecolor("#0f172a"); ax_r1.set_facecolor("#1e293b")
            col_r1b = ["#34d399" if v>=0 else "#f43f5e" for v in contribs]
            ax_r1.bar(feat_labels, contribs, color=col_r1b, edgecolor="#0f172a")
            ax_r1.axhline(0, color="#94a3b8", linewidth=1)
            ax_r1.set_title(f"Contributions — Candidate #{int(top_row['cand_id'])}",
                            color="white", fontsize=11)
            ax_r1.tick_params(axis="x", rotation=20, colors="#94a3b8")
            ax_r1.tick_params(axis="y", colors="#94a3b8")
            ax_r1.set_ylabel("Weighted Contribution", color="#94a3b8")
            st.pyplot(fig_r1)

    with col_r4:
        st.subheader("MMR Re-ranking")
        st.latex(r"\text{MMR}=\arg\max_{d_i\in R\setminus S}"
                 r"\!\left[\lambda\cdot\text{sim}(d_i,q)-(1-\lambda)\cdot"
                 r"\max_{d_j\in S}\text{sim}(d_i,d_j)\right]")
        full_c   = score_candidates(sel_job_row, cands_df, W, B, mu, sig, skill_w, exp_w)
        mmr_c    = mmr_rerank(full_c, lam=mmr_lam, top_n=10)
        orig10   = full_c.head(10)["cand_id"].tolist()
        mmr10    = mmr_c["cand_id"].tolist()
        changed  = sum(1 for a,b2 in zip(orig10, mmr10) if a!=b2)
        st.info(f"λ={mmr_lam:.2f} → **{changed}/10 positions changed** vs greedy")
        st.dataframe(mmr_c[["cand_id","current_role","skill_overlap_pct","score"]],
                     use_container_width=True, height=260)

    st.subheader("Salary Alignment — Top Candidates vs Job Range")
    if len(ranked_c):
        top10_cids = ranked_c.head(10)["cand_id"].astype(int).tolist()
        top10_sals = [cands_df.loc[cid,"current_salary"] for cid in top10_cids]
        fig_r2, ax_r2 = plt.subplots(figsize=(10,3.5))
        fig_r2.patch.set_facecolor("#0f172a"); ax_r2.set_facecolor("#1e293b")
        ax_r2.axhspan(sel_job_row["salary_min"], sel_job_row["salary_max"],
                      alpha=0.2, color="#34d399", label="Job Salary Band")
        ax_r2.plot(range(len(top10_sals)), top10_sals, "o",
                   color="#38bdf8", markersize=9, label="Candidate Salary")
        ax_r2.axhline(sel_job_row["salary_min"], color="#34d399", linestyle="--", alpha=0.6)
        ax_r2.axhline(sel_job_row["salary_max"], color="#34d399", linestyle="--", alpha=0.6)
        ax_r2.set_xticks(range(len(top10_sals)))
        ax_r2.set_xticklabels([f"#{c}" for c in top10_cids], rotation=30, color="#94a3b8")
        ax_r2.tick_params(axis="y", colors="#94a3b8")
        ax_r2.set_ylabel("Salary ($)", color="#94a3b8")
        ax_r2.set_title("Candidate Salary vs Job Range", color="white", fontsize=12)
        ax_r2.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
        ax_r2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
        st.pyplot(fig_r2)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📊 Market Intelligence — Talent Analytics")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Skill Demand vs Supply Gap (Top 15)**")
        t15n = [s for s,_ in sk_cnt.most_common(15)]
        d15  = [sk_cnt[s] for s in t15n]
        s15  = [sum(1 for r in cands_df["skills"] for sk2 in r if sk2==s) for s in t15n]
        fig_m1, ax_m1 = plt.subplots(figsize=(7,5))
        fig_m1.patch.set_facecolor("#0f172a"); ax_m1.set_facecolor("#1e293b")
        xm1 = np.arange(len(t15n))
        ax_m1.bar(xm1-0.2, d15, 0.35, label="Job Demand",       color="#818cf8", alpha=0.9)
        ax_m1.bar(xm1+0.2, s15, 0.35, label="Candidate Supply", color="#34d399", alpha=0.9)
        ax_m1.set_xticks(xm1); ax_m1.set_xticklabels(t15n, rotation=40, color="#94a3b8", ha="right")
        ax_m1.tick_params(axis="y",colors="#94a3b8")
        ax_m1.set_title("Skill Demand vs Supply", color="white", fontsize=12)
        ax_m1.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
        st.pyplot(fig_m1)

    with col_m2:
        st.markdown("**Salary Benchmarking by Job Title**")
        fig_m2, ax_m2 = plt.subplots(figsize=(7,5))
        fig_m2.patch.set_facecolor("#0f172a"); ax_m2.set_facecolor("#1e293b")
        bp_data = [jobs_df.loc[jobs_df["title"]==t,"salary_max"].values for t in JOB_TITLES]
        bp = ax_m2.boxplot(bp_data, patch_artist=True, notch=False)
        for patch, col in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        for el in ["whiskers","caps","medians"]:
            for line in bp[el]: line.set_color("#94a3b8")
        for fl in bp["fliers"]: fl.set(marker="o",color="#f472b6",alpha=0.5)
        ax_m2.set_xticklabels(JOB_TITLES, rotation=20, color="#94a3b8", ha="right")
        ax_m2.tick_params(axis="y",colors="#94a3b8")
        ax_m2.set_title("Salary Max by Job Title", color="white", fontsize=12)
        ax_m2.set_ylabel("Salary ($)",color="#94a3b8")
        ax_m2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
        st.pyplot(fig_m2)

    st.markdown("---")
    col_m3, col_m4 = st.columns(2)

    with col_m3:
        st.markdown("**Hiring Funnel Analytics (Simulated)**")
        rng_f   = np.random.default_rng(7)
        dec     = np.arange(0.05, 1.05, 0.1)
        ar      = np.clip(dec*1.2  + rng_f.normal(0,0.05,len(dec)), 0, 1)
        ir      = np.clip(ar*dec   + rng_f.normal(0,0.03,len(dec)), 0, 1)
        ofr     = np.clip(ir*dec*1.5 + rng_f.normal(0,0.02,len(dec)), 0, 1)
        fig_m3, ax_m3 = plt.subplots(figsize=(7,4))
        fig_m3.patch.set_facecolor("#0f172a"); ax_m3.set_facecolor("#1e293b")
        xd = np.arange(len(dec))
        ax_m3.plot(xd, ar,  "o-", color="#38bdf8", linewidth=2, label="Apply Rate")
        ax_m3.plot(xd, ir,  "s-", color="#34d399", linewidth=2, label="Interview Rate")
        ax_m3.plot(xd, ofr, "^-", color="#fb923c", linewidth=2, label="Offer Rate")
        ax_m3.set_xticks(xd)
        ax_m3.set_xticklabels([f"{d:.0%}" for d in dec], rotation=30, color="#94a3b8")
        ax_m3.tick_params(axis="y",colors="#94a3b8")
        ax_m3.set_title("Hiring Funnel by Match Score Decile", color="white", fontsize=11)
        ax_m3.set_xlabel("Match Score Decile",color="#94a3b8")
        ax_m3.set_ylabel("Rate",color="#94a3b8")
        ax_m3.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
        ax_m3.set_ylim(0,1.05)
        st.pyplot(fig_m3)

    with col_m4:
        st.markdown("**Time-to-Fill: Qualified Candidates per Job (sample 30 jobs)**")
        thresholds = [0.50, 0.70, 0.90]
        ttf_rows   = []
        for ji in range(min(30, len(jobs_df))):
            rc = score_candidates(jobs_df.loc[ji], cands_df, W, B, mu, sig)
            for thr in thresholds:
                ttf_rows.append({"threshold": thr,
                                 "qualified": int((rc["score"]>=thr).sum())})
        ttf_df = pd.DataFrame(ttf_rows).groupby("threshold")["qualified"].mean().reset_index()
        fig_m4, ax_m4 = plt.subplots(figsize=(6,4))
        fig_m4.patch.set_facecolor("#0f172a"); ax_m4.set_facecolor("#1e293b")
        brs = ax_m4.bar([f"≥{t:.0%}" for t in ttf_df["threshold"]],
                        ttf_df["qualified"], color=["#34d399","#facc15","#f43f5e"])
        ax_m4.set_title("Avg Qualified Candidates per Job\n(by Threshold)",
                        color="white", fontsize=12)
        ax_m4.tick_params(colors="#94a3b8")
        ax_m4.set_ylabel("Avg Candidates",color="#94a3b8")
        for br, v in zip(brs, ttf_df["qualified"]):
            ax_m4.text(br.get_x()+br.get_width()/2, v+0.3,
                       f"{v:.1f}", ha="center", color="white", fontsize=10)
        st.pyplot(fig_m4)

    st.markdown("---")
    col_m5, col_m6 = st.columns(2)

    with col_m5:
        st.markdown("**Trending Skills — Simulated YoY Growth (%)**")
        rng_t = np.random.default_rng(99)
        t15   = [s for s,_ in sk_cnt.most_common(15)]
        yoy   = rng_t.normal(0, 12, len(t15))
        for i, s in enumerate(t15):
            if s in ("ML","TensorFlow","PyTorch","AWS","Docker","NLP","Kubernetes"):
                yoy[i] += rng_t.uniform(8, 20)
            elif s in ("Excel","Tableau"):
                yoy[i] -= rng_t.uniform(5, 12)
        fig_m5, ax_m5 = plt.subplots(figsize=(7,4))
        fig_m5.patch.set_facecolor("#0f172a"); ax_m5.set_facecolor("#1e293b")
        c_yoy = ["#34d399" if v>0 else "#f43f5e" for v in yoy]
        ax_m5.barh(t15, yoy, color=c_yoy)
        ax_m5.axvline(0, color="#94a3b8", linewidth=1)
        ax_m5.set_title("Skill YoY Demand Growth (%)", color="white", fontsize=12)
        ax_m5.tick_params(colors="#94a3b8")
        ax_m5.set_xlabel("YoY Change (%)",color="#94a3b8")
        st.pyplot(fig_m5)

    with col_m6:
        st.markdown("**Geographic Distribution — Jobs vs Candidate Preferences**")
        job_loc  = jobs_df["location"].value_counts().reindex(LOCATIONS, fill_value=0)
        cand_loc = pd.Series({
            loc: sum(1 for r in cands_df["preferred_locations"] for l in r if l==loc)
            for loc in LOCATIONS})
        fig_m6, ax_m6 = plt.subplots(figsize=(7,4))
        fig_m6.patch.set_facecolor("#0f172a"); ax_m6.set_facecolor("#1e293b")
        xg = np.arange(len(LOCATIONS))
        ax_m6.bar(xg-0.2, job_loc.values,  0.35, label="Jobs",       color="#38bdf8", alpha=0.9)
        ax_m6.bar(xg+0.2, cand_loc.values, 0.35, label="Candidates", color="#f472b6", alpha=0.9)
        ax_m6.set_xticks(xg); ax_m6.set_xticklabels(LOCATIONS, rotation=20, color="#94a3b8")
        ax_m6.tick_params(axis="y",colors="#94a3b8")
        ax_m6.set_title("Geographic Distribution", color="white", fontsize=12)
        ax_m6.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
        st.pyplot(fig_m6)

    st.subheader("Market Tightness — Demand/Supply Ratio by Skill Group")
    tightness = {}
    for grp, sk in SKILL_GROUPS.items():
        d = sum(1 for r in jobs_df["required_skills"] for s in r if s in sk)
        s_v = sum(1 for r in cands_df["skills"] for s in r if s in sk)
        tightness[grp] = d / max(s_v, 1)
    tight = pd.DataFrame({"Group": list(tightness.keys()),
                           "Ratio": list(tightness.values())}).sort_values("Ratio", ascending=False)
    fig_mt, ax_mt = plt.subplots(figsize=(10,3))
    fig_mt.patch.set_facecolor("#0f172a"); ax_mt.set_facecolor("#1e293b")
    ct = ["#f43f5e" if v>0.65 else "#facc15" if v>0.45 else "#34d399" for v in tight["Ratio"]]
    brs_t = ax_mt.bar(tight["Group"], tight["Ratio"], color=ct)
    ax_mt.axhline(1.0, color="#94a3b8", linestyle="--", alpha=0.7, label="Balanced")
    for br, v in zip(brs_t, tight["Ratio"]):
        ax_mt.text(br.get_x()+br.get_width()/2, v+0.01,
                   f"{v:.2f}x", ha="center", color="white", fontsize=10)
    ax_mt.set_title("Market Tightness (>1 = undersupplied)", color="white", fontsize=12)
    ax_mt.tick_params(colors="#94a3b8"); ax_mt.set_ylabel("D/S Ratio",color="#94a3b8")
    ax_mt.legend(facecolor="#1e293b",edgecolor="#334155",labelcolor="white")
    st.pyplot(fig_mt)

    st.caption("JobMatch · AI Talent Intelligence Platform · NumPy + Streamlit · No sklearn")
