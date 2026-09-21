import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   
from data_loader import load_data
from predict import load_model, predict

st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("Salary Predictor")
st.caption("ML model trained on 250K job records | scikit-learn + Streamlit")

@st.cache_resource
def get_model():
    return load_model()

pipe = get_model()

@st.cache_data(ttl=3600)
def get_reference_data():
    return load_data()

df = get_reference_data()

# Get unique values
job_titles = sorted(df["job_title"].unique().tolist())
educations = sorted(df["education_level"].unique().tolist())
industries = sorted(df["industry"].unique().tolist())
company_sizes = sorted(df["company_size"].unique().tolist())
locations = sorted(df["location"].unique().tolist())
remote_options = sorted(df["remote_work"].unique().tolist())

# --- Sidebar: Input Form ---
st.sidebar.header("Enter Your Details")

job_title = st.sidebar.selectbox("Job Title", job_titles)
exp_years = st.sidebar.slider("Years of Experience", 0, 40, 5)
education = st.sidebar.selectbox("Education Level", educations)
skills_count = st.sidebar.slider("Number of Skills", 0, 50, 10)
industry = st.sidebar.selectbox("Industry", industries)
company_size = st.sidebar.selectbox("Company Size", company_sizes)
location = st.sidebar.selectbox("Location", locations)
remote = st.sidebar.selectbox("Remote Work", remote_options)
certs = st.sidebar.slider("Number of Certifications", 0, 10, 1)

features = {
    "job_title": job_title,
    "experience_years": exp_years,
    "education_level": education,
    "skills_count": skills_count,
    "industry": industry,
    "company_size": company_size,
    "location": location,
    "remote_work": remote,
    "certifications": certs,
}

# --- PREDICT ---
salary = predict(pipe, features)

# --- BENCHMARKS (the "explanation" part) ---
# Overall stats
overall_median = df["salary"].median()

# Same job title
same_job = df[df["job_title"] == job_title]
job_median = same_job["salary"].median() if len(same_job) > 0 else overall_median
job_p25 = same_job["salary"].quantile(0.25) if len(same_job) > 0 else 0
job_p75 = same_job["salary"].quantile(0.75) if len(same_job) > 0 else 0   

# Same job + location
same_job_loc = df[(df["job_title"] == job_title) & (df["location"] == location)]
job_loc_median = same_job_loc["salary"].median() if len(same_job_loc) > 0 else job_median

# Percentile in same job
percentile = (same_job["salary"] < salary).mean() * 100 if len(same_job) > 0 else 50

# --- LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Prediction")
    st.markdown(f"### **${salary:,.0f}** / year")
    st.markdown(f"*{job_title} | {location} | {exp_years} yrs exp*")

    st.divider()
    st.subheader("Benchmarks")
    st.markdown(f"| Metric | Value |")
    st.markdown(f"|--------|-------|")
    st.markdown(f"| Your prediction | **${salary:,.0f}** |")
    st.markdown(f"| Median ({job_title}) | ${job_median:,.0f} |")
    st.markdown(f"| Median ({job_title} in {location}) | ${job_loc_median:,.0f} |")
    st.markdown(f"| Overall median | ${overall_median:,.0f} |")

    st.divider()
    pct_label = "above" if salary > job_median else "below"
    pct_diff = abs(salary - job_median) / job_median * 100 if job_median > 0 else 0
    st.markdown(f"### {percentile:.0f}th percentile")
    st.markdown(f"Your prediction is **{pct_label} the median** for {job_title}s by **${abs(salary - job_median):,.0f}** ({pct_diff:.0f}%).")

with col2:
    st.subheader("Salary Distribution for " + job_title)
    
    # Plot: distribution with your prediction marked
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Histogram of same-job salaries
    ax.hist(same_job["salary"], bins=40, color="#4A90D9", alpha=0.6, 
            edgecolor="white", label=f"{job_title} (n={len(same_job):,})")
    
    # Mark your prediction
    ax.axvline(x=salary, color="red", linewidth=2, linestyle="--", label=f"You: ${salary:,.0f}")
    # Mark median
    ax.axvline(x=job_median, color="green", linewidth=1.5, label=f"Median: ${job_median:,.0f}")
    # Mark 25th and 75th
    ax.axvline(x=job_p25, color="orange", linewidth=1, linestyle=":", label=f"25th: ${job_p25:,.0f}")
    ax.axvline(x=job_p75, color="orange", linewidth=1, linestyle=":", label=f"75th: ${job_p75:,.0f}")
    
    ax.set_xlabel("Annual Salary ($)")
    ax.set_ylabel("Number of Jobs")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Where you fall among {len(same_job):,} {job_title}s")
    
    st.pyplot(fig)

# --- "WHAT IF" SECTION ---
st.divider()
st.subheader("🔍 What-If Analysis")
st.caption("See how changing one factor affects your prediction (all else stays the same)")

what_if = st.radio("Change which factor?", ["Experience", "Location", "Company Size", "Remote Work", "Education"], horizontal=True)

if what_if == "Experience":
    exp_range = list(range(0, 41))
    predictions = []
    for e in exp_range:
        f = features.copy()
        f["experience_years"] = e
        predictions.append(predict(pipe, f))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(exp_range, predictions, color="#4A90D9", linewidth=2)
    ax.fill_between(exp_range, predictions, alpha=0.1, color="#4A90D9")
    ax.axvline(x=exp_years, color="red", linewidth=1.5, linestyle="--", label=f"Your level: {exp_years} yrs")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Predicted Salary ($)")
    ax.set_title(f"Salary vs Experience ({job_title}, {location})")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    st.pyplot(fig)

elif what_if == "Location":
    loc_predictions = []
    loc_names = []
    for loc in locations:
        f = features.copy()
        f["location"] = loc
        loc_predictions.append(predict(pipe, f))
        loc_names.append(loc)
    
    loc_df = pd.DataFrame({"location": loc_names, "salary": loc_predictions})
    loc_df = loc_df.sort_values("salary", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["red" if l == location else "#4A90D9" for l in loc_df["location"]]
    ax.barh(loc_df["location"], loc_df["salary"], color=colors)
    ax.set_xlabel("Predicted Salary ($)")
    ax.set_title(f"Salary by Location ({job_title})")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    st.pyplot(fig)

elif what_if == "Company Size":
    size_predictions = []
    for cs in company_sizes:
        f = features.copy()
        f["company_size"] = cs
        size_predictions.append(predict(pipe, f))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["red" if c == company_size else "#4A90D9" for c in company_sizes]
    ax.bar(company_sizes, size_predictions, color=colors)
    ax.set_ylabel("Predicted Salary ($)")
    ax.set_title(f"Salary by Company Size ({job_title}, {location})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)

elif what_if == "Remote Work":
    remote_predictions = []
    for r in remote_options:
        f = features.copy()
        f["remote_work"] = r
        remote_predictions.append(predict(pipe, f))
    
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["red" if r == remote else "#4A90D9" for r in remote_options]
    ax.bar(remote_options, remote_predictions, color=colors)
    ax.set_ylabel("Predicted Salary ($)")
    ax.set_title(f"Salary by Remote Status ({job_title}, {location})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    st.pyplot(fig)

elif what_if == "Education":
    edu_predictions = []
    for ed in educations:
        f = features.copy()
        f["education_level"] = ed
        edu_predictions.append(predict(pipe, f))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["red" if e == education else "#4A90D9" for e in educations]
    ax.bar(educations, edu_predictions, color=colors)
    ax.set_ylabel("Predicted Salary ($)")
    ax.set_title(f"Salary by Education Level ({job_title}, {location})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)

# --- Footer ---
st.divider()
st.caption("Model: Random Forest (R²=0.97) | Trained on 250K records | [GitHub](https://github.com/YOUR_USERNAME/salary-predictor)")   
