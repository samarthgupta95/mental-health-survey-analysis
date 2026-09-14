import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import streamlit as st

st.set_page_config(page_title="Mental Health in Tech — Dashboard", layout="wide")
sns.set_style("whitegrid")


@st.cache_data
def load_data():
    return pd.read_csv("survey_clean.csv")


df = load_data()

st.title("Mental Health in Tech Survey — Interactive Dashboard")
st.caption("OSMI 2014 survey · 1,251 respondents after cleaning · exploring what drives treatment-seeking at work")

# --- sidebar filters ---
st.sidebar.header("Filters")

countries = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect("Country", countries, default=[])

genders = sorted(df["Gender"].unique())
selected_genders = st.sidebar.multiselect("Gender", genders, default=[])

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

size_order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
selected_sizes = st.sidebar.multiselect("Company size", size_order, default=[])

tech_only = st.sidebar.selectbox("Company type", ["All", "Tech companies only", "Non-tech only"])

filtered = df.copy()
if selected_countries:
    filtered = filtered[filtered["Country"].isin(selected_countries)]
if selected_genders:
    filtered = filtered[filtered["Gender"].isin(selected_genders)]
filtered = filtered[filtered["Age"].between(*age_range)]
if selected_sizes:
    filtered = filtered[filtered["no_employees"].isin(selected_sizes)]
if tech_only == "Tech companies only":
    filtered = filtered[filtered["tech_company"] == "Yes"]
elif tech_only == "Non-tech only":
    filtered = filtered[filtered["tech_company"] == "No"]

st.sidebar.markdown(f"**{len(filtered)}** respondents match these filters")

if len(filtered) < 10:
    st.warning("Fewer than 10 respondents match this filter combination — treat the numbers below as illustrative only.")

# --- top metrics ---
col1, col2, col3, col4 = st.columns(4)
treatment_rate = (filtered["treatment"] == "Yes").mean() * 100 if len(filtered) else 0
family_yes_rate = (
    filtered.loc[filtered["family_history"] == "Yes", "treatment"].eq("Yes").mean() * 100
    if (filtered["family_history"] == "Yes").any() else 0
)
care_yes_rate = (
    filtered.loc[filtered["care_options"] == "Yes", "treatment"].eq("Yes").mean() * 100
    if (filtered["care_options"] == "Yes").any() else 0
)
benefits_yes_rate = (
    filtered.loc[filtered["benefits"] == "Yes", "treatment"].eq("Yes").mean() * 100
    if (filtered["benefits"] == "Yes").any() else 0
)

col1.metric("Treatment-seeking rate", f"{treatment_rate:.1f}%")
col2.metric("...with family history", f"{family_yes_rate:.1f}%")
col3.metric("...with clear care options", f"{care_yes_rate:.1f}%")
col4.metric("...with mental health benefits", f"{benefits_yes_rate:.1f}%")

st.divider()

# --- demographics row ---
st.subheader("Who's in this view")
d1, d2, d3 = st.columns(3)

with d1:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(filtered["Age"], bins=20, ax=ax, color="#4C72B0")
    ax.set_title("Age distribution")
    st.pyplot(fig)

with d2:
    fig, ax = plt.subplots(figsize=(4, 3))
    gender_counts = filtered["Gender"].value_counts()
    sns.barplot(x=gender_counts.values, y=gender_counts.index, ax=ax, color="#55A868")
    ax.set_title("Gender split")
    st.pyplot(fig)

with d3:
    fig, ax = plt.subplots(figsize=(4, 3))
    top_countries = filtered["Country"].value_counts().head(8)
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax, color="#C44E52")
    ax.set_title("Top countries")
    st.pyplot(fig)

st.divider()

# --- predictors ---
st.subheader("What predicts treatment-seeking")
st.caption("Cramér's V association strength between each field and treatment-seeking, computed on the currently filtered data")


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))


candidate_cols = ["family_history", "work_interfere", "benefits", "care_options", "wellness_program",
                   "seek_help", "anonymity", "leave", "mental_health_consequence", "phys_health_consequence",
                   "coworkers", "supervisor", "mental_health_interview", "phys_health_interview",
                   "mental_vs_physical", "obs_consequence", "no_employees", "remote_work", "tech_company",
                   "self_employed", "Gender"]

scores = {}
for col in candidate_cols:
    sub = filtered[[col, "treatment"]].dropna()
    if len(sub) >= 10:
        scores[col] = cramers_v(sub[col], sub["treatment"])

if scores:
    score_series = pd.Series(scores).dropna().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=score_series.values, y=score_series.index, ax=ax, color="#8172B2")
    ax.set_xlabel("Cramér's V")
    ax.set_title("Predictor strength vs. treatment-seeking")
    st.pyplot(fig)
else:
    st.info("Not enough data in this filter combination to compute associations.")

st.divider()

# --- employer levers detail ---
st.subheader("Employer-controllable levers")
l1, l2 = st.columns(2)

with l1:
    rate = filtered.groupby("care_options")["treatment"].apply(lambda s: (s == "Yes").mean() * 100)
    rate = rate.reindex(["No", "Not sure", "Yes"]).dropna()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#4C72B0")
    ax.set_ylabel("Treatment rate (%)")
    ax.set_title("By care options awareness")
    st.pyplot(fig)

with l2:
    rate = filtered.groupby("benefits")["treatment"].apply(lambda s: (s == "Yes").mean() * 100)
    rate = rate.reindex(["No", "Don't know", "Yes"]).dropna()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#DD8452")
    ax.set_ylabel("Treatment rate (%)")
    ax.set_title("By mental health benefits")
    st.pyplot(fig)

st.divider()

# --- geographic cut ---
st.subheader("Geographic patterns")
country_counts = filtered["Country"].value_counts()
reliable = country_counts[country_counts >= 10].index
if len(reliable) > 0:
    geo = filtered[filtered["Country"].isin(reliable)].groupby("Country")["treatment"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, len(geo) * 0.4)))
    sns.barplot(x=geo.values, y=geo.index, ax=ax, color="#55A868")
    ax.set_xlabel("Treatment rate (%)")
    ax.set_title("Treatment rate by country (n >= 10 respondents in current filter)")
    st.pyplot(fig)
else:
    st.info("No country has enough respondents (n >= 10) in this filter combination.")

st.divider()
st.caption("Source: OSMI Mental Health in Tech Survey (2014), cleaned for age outliers and gender free-text.")
