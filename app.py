import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mental Health in Tech", layout="wide")

# ---------- palette ----------
BG = "#12161D"
SURFACE = "#1B212B"
BORDER = "#2A313D"
TEXT = "#E7E9ED"
MUTED = "#8B93A3"
TEAL = "#4FA8A0"
GOLD = "#E3A868"
ROSE = "#C97B84"
FONT = "Inter, -apple-system, sans-serif"
DISPLAY_FONT = "'Fraunces', Georgia, serif"

# ---------- styling ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT};
}}

.block-container {{
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

.hero-title {{
    font-family: {DISPLAY_FONT};
    font-weight: 500;
    font-size: 2.6rem;
    color: {TEXT};
    line-height: 1.15;
    margin-bottom: 0.4rem;
}}

.hero-sub {{
    font-family: {FONT};
    font-size: 1.02rem;
    color: {MUTED};
    max-width: 640px;
    line-height: 1.5;
    margin-bottom: 1.6rem;
}}

.section-title {{
    font-family: {DISPLAY_FONT};
    font-weight: 500;
    font-size: 1.5rem;
    color: {TEXT};
    margin-top: 0.2rem;
    margin-bottom: 0.3rem;
}}

.section-note {{
    font-family: {FONT};
    font-size: 0.92rem;
    color: {MUTED};
    margin-bottom: 1rem;
    max-width: 720px;
    line-height: 1.5;
}}

hr.divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 2.2rem 0 1.6rem 0;
}}

.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid var(--accent, {TEAL});
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}}

.kpi-label {{
    font-family: {FONT};
    font-size: 0.85rem;
    color: {MUTED};
    margin-bottom: 0.35rem;
}}

.kpi-value {{
    font-family: {DISPLAY_FONT};
    font-weight: 500;
    font-size: 2.1rem;
    color: {TEXT};
}}

.callout {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {GOLD};
    border-radius: 6px;
    padding: 1rem 1.3rem;
    font-family: {FONT};
    font-size: 0.92rem;
    color: {TEXT};
    line-height: 1.55;
    margin-top: 0.6rem;
}}

section[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER};
}}
</style>
""", unsafe_allow_html=True)


def chart_layout(fig, height=340):
    current_title = fig.layout.title.text if fig.layout.title is not None else None
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12.5),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        title=dict(text=current_title or "", font=dict(family=FONT, size=14, color=TEXT)),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def kpi_card(label, value, accent):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title, note=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("survey_clean.csv")
    bins = [17, 24, 34, 44, 54, 100]
    labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
    df["age_band"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


df = load_data()

# ---------- sidebar filters ----------
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

# ---------- hero ----------
st.markdown('<div class="hero-title">Mental Health in Tech</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Based on the 2014 OSMI survey of 1,251 tech employees. '
    'An interactive look at what actually drives treatment-seeking at work, and which '
    'of it an employer can influence.</div>',
    unsafe_allow_html=True,
)

if len(filtered) < 10:
    st.warning("Fewer than 10 respondents match this filter combination. Treat the numbers below as illustrative only.")

# ---------- kpis ----------
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

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Treatment-seeking rate", f"{treatment_rate:.1f}%", TEAL)
with k2:
    kpi_card("With family history", f"{family_yes_rate:.1f}%", ROSE)
with k3:
    kpi_card("With clear care options", f"{care_yes_rate:.1f}%", GOLD)
with k4:
    kpi_card("With mental health benefits", f"{benefits_yes_rate:.1f}%", GOLD)

divider()

# ---------- demographics ----------
section("Who's in this view")
d1, d2, d3 = st.columns(3)

with d1:
    fig = px.histogram(filtered, x="Age", nbins=20)
    fig.update_traces(marker_color=TEAL, marker_line_width=0)
    fig.update_layout(title="Age distribution")
    st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})

with d2:
    gc = filtered["Gender"].value_counts()
    fig = px.bar(x=gc.values, y=gc.index, orientation="h")
    fig.update_traces(marker_color=TEAL, marker_line_width=0)
    fig.update_layout(title="Gender split", yaxis_title=None, xaxis_title=None)
    st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})

with d3:
    tc = filtered["Country"].value_counts().head(8).sort_values()
    fig = px.bar(x=tc.values, y=tc.index, orientation="h")
    fig.update_traces(marker_color=TEAL, marker_line_width=0)
    fig.update_layout(title="Top countries", yaxis_title=None, xaxis_title=None)
    st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})

divider()

# ---------- predictors ----------
section(
    "What predicts treatment-seeking",
    "Cramér's V association strength between each field and treatment-seeking, on the currently filtered data. Higher means the two move together more strongly.",
)


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
        v = cramers_v(sub[col], sub["treatment"])
        if not np.isnan(v):
            scores[col] = v

if scores:
    s = pd.Series(scores).sort_values()
    s.index = s.index.str.replace("_", " ")
    fig = px.bar(x=s.values, y=s.index, orientation="h", color=s.values, color_continuous_scale=[SURFACE, TEAL])
    fig.update_layout(xaxis_title="Cramér's V", yaxis_title=None, coloraxis_showscale=False)
    st.plotly_chart(chart_layout(fig, 460), width="stretch", config={"displayModeBar": False})
else:
    st.info("Not enough data in this filter combination to compute associations.")

divider()

# ---------- employer levers ----------
section(
    "Employer-controllable levers",
    "Of everything measured, these are the factors a company can actually change. Bars show the treatment-seeking rate for each response category.",
)

lever_specs = [
    ("care_options", ["No", "Not sure", "Yes"]),
    ("benefits", ["No", "Don't know", "Yes"]),
    ("wellness_program", ["No", "Don't know", "Yes"]),
    ("anonymity", ["No", "Don't know", "Yes"]),
    ("leave", ["Very difficult", "Somewhat difficult", "Don't know", "Somewhat easy", "Very easy"]),
]

lever_cols = st.columns(len(lever_specs))
for col, (field, order) in zip(lever_cols, lever_specs):
    with col:
        rate = filtered.groupby(field)["treatment"].apply(lambda s: (s == "Yes").mean() * 100)
        rate = rate.reindex(order).dropna()
        fig = px.bar(x=rate.index, y=rate.values)
        fig.update_traces(marker_color=GOLD, marker_line_width=0)
        fig.update_layout(title=field.replace("_", " "), xaxis_title=None, yaxis_title=None)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})

st.markdown(
    '<div class="callout">Leave doesn\'t move in a straight line: people who say taking leave is '
    '"very difficult" seek treatment at a higher rate than those who say it\'s easy. That\'s likely '
    'the arrow running backwards, people already managing a condition notice friction they\'d '
    'otherwise not have tested, rather than hard leave policy causing more treatment.</div>',
    unsafe_allow_html=True,
)

divider()

# ---------- openness ----------
section(
    "Talking about it at work",
    "How willing are respondents to discuss a mental health issue in different settings? Distribution of responses, not treatment rate.",
)

o1, o2, o3 = st.columns(3)
openness_specs = [
    ("coworkers", "With coworkers", o1),
    ("supervisor", "With a supervisor", o2),
    ("mental_health_interview", "In a job interview", o3),
]
for field, label, col in openness_specs:
    with col:
        counts = filtered[field].value_counts(normalize=True).mul(100)
        order = [c for c in ["Yes", "Some of them", "Maybe", "No"] if c in counts.index]
        counts = counts.reindex(order)
        fig = px.bar(x=counts.index, y=counts.values)
        fig.update_traces(marker_color=ROSE, marker_line_width=0)
        fig.update_layout(title=label, xaxis_title=None, yaxis_title="% of respondents")
        st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})

st.markdown(
    '<div class="callout">Willingness drops off sharply as the setting becomes more formal: '
    '61.6% would tell some coworkers and 40.9% would tell a supervisor, but only 3.3% would '
    'raise it in a job interview and 80.2% flatly would not. The stigma isn\'t evenly distributed, '
    'it concentrates at the moments with the most at stake.</div>',
    unsafe_allow_html=True,
)

divider()

# ---------- age ----------
section("Does age change the picture?", "Treatment-seeking rate by age band. Sample sizes shrink sharply past 45, shown as labels above each bar.")

age_counts = filtered["age_band"].value_counts()
age_rate = filtered.groupby("age_band", observed=True)["treatment"].apply(lambda s: (s == "Yes").mean() * 100)
age_order = ["18-24", "25-34", "35-44", "45-54", "55+"]
age_rate = age_rate.reindex([a for a in age_order if a in age_rate.index])

fig = go.Figure(go.Bar(
    x=age_rate.index.astype(str), y=age_rate.values,
    marker_color=TEAL,
    text=[f"n={age_counts.get(a, 0)}" for a in age_rate.index],
    textposition="outside",
    textfont=dict(color=MUTED, size=11),
))
fig.update_layout(yaxis_title="Treatment rate (%)")
st.plotly_chart(chart_layout(fig, 340), width="stretch", config={"displayModeBar": False})

divider()

# ---------- geography ----------
section("Geographic patterns", "Restricted to countries with at least 10 respondents in the current filter, for reliability.")

country_counts = filtered["Country"].value_counts()
reliable = country_counts[country_counts >= 10].index
if len(reliable) > 0:
    geo = filtered[filtered["Country"].isin(reliable)].groupby("Country")["treatment"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).sort_values()
    fig = px.bar(x=geo.values, y=geo.index, orientation="h", color=geo.values, color_continuous_scale=[SURFACE, TEAL])
    fig.update_layout(xaxis_title="Treatment rate (%)", yaxis_title=None, coloraxis_showscale=False)
    st.plotly_chart(chart_layout(fig, max(280, len(geo) * 34)), width="stretch", config={"displayModeBar": False})
else:
    st.info("No country has enough respondents (n >= 10) in this filter combination.")

divider()
st.markdown(
    f'<div style="color:{MUTED}; font-size:0.85rem;">Source: OSMI Mental Health in Tech Survey (2014). '
    f'Cleaned for age outliers and gender free-text before analysis.</div>',
    unsafe_allow_html=True,
)
