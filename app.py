import html

import streamlit as st
import pandas as pd

from benchmark import (
    prepare_dataset,
    split_features_target,
    run_benchmark,
    find_best_combination,
)

st.set_page_config(
    page_title="PreprocessIQ",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens
#   ink     #0F1B3D  deep navy, hero + headings
#   aqua    #14B8A6  primary accent, buttons, highlights
#   sun     #FFC53D  secondary accent, best-result badge
#   cloud   #F4F7FB  page background
#   slate   #5B6785  body text
# Fonts: Outfit (headings) + DM Sans (body)
# ---------------------------------------------------------------------------

HERO_IMAGE = (
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71"
    "?auto=format&fit=crop&w=1800&q=80"
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Outfit:wght@500;600;700;800&display=swap');

:root {{
    --ink: #0F1B3D;
    --aqua: #14B8A6;
    --aqua-dark: #0E8F81;
    --sun: #FFC53D;
    --cloud: #F4F7FB;
    --slate: #5B6785;
    --line: #DDE4EF;
}}

html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', sans-serif;
    color: var(--slate);
}}

.stApp {{
    background: var(--cloud);
}}

#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}}

h1, h2, h3, h4 {{
    font-family: 'Outfit', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}}

/* ---------- Hero ---------- */
.hero {{
    position: relative;
    border-radius: 28px;
    overflow: hidden;
    padding: 64px 56px;
    margin-bottom: 28px;
    background:
        linear-gradient(105deg, rgba(15,27,61,0.96) 0%, rgba(15,27,61,0.86) 48%, rgba(20,184,166,0.45) 100%),
        url('{HERO_IMAGE}') center/cover no-repeat,
        var(--ink);
    box-shadow: 0 24px 60px -24px rgba(15,27,61,0.55);
}}
.hero h1 {{
    font-family: 'Outfit', sans-serif !important;
    color: #FFFFFF !important;
    font-size: 3.4rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    line-height: 1.05;
}}
.hero h1 .iq {{
    color: var(--sun) !important;
}}
.hero .tagline {{
    font-family: 'Outfit', sans-serif;
    color: #BFEFE9;
    font-size: 1.35rem;
    font-weight: 500;
    margin-bottom: 18px;
}}
.hero p {{
    color: #D6DEF0;
    max-width: 620px;
    font-size: 1.05rem;
    line-height: 1.65;
    margin: 0;
}}

/* ---------- Step cards ---------- */
.steps {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 30px;
}}
.step {{
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px 20px 18px 20px;
}}
.step .icon {{
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    margin-bottom: 12px;
}}
.step:nth-child(1) .icon {{ background: #DDF7F3; }}
.step:nth-child(2) .icon {{ background: #FFF1CC; }}
.step:nth-child(3) .icon {{ background: #E3E8FB; }}
.step:nth-child(4) .icon {{ background: #FFE1E1; }}
.step h4 {{
    margin: 0 0 4px 0;
    font-size: 1.05rem;
}}
.step p {{
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
}}

/* ---------- Section headings ---------- */
.section-title {{
    font-family: 'Outfit', sans-serif;
    color: var(--ink);
    font-size: 1.55rem;
    font-weight: 700;
    margin: 34px 0 12px 0;
    padding-left: 14px;
    border-left: 5px solid var(--aqua);
    line-height: 1.2;
}}

/* ---------- Upload ---------- */
[data-testid="stFileUploaderDropzone"] {{
    background: #FFFFFF;
    border: 2px dashed var(--aqua);
    border-radius: 20px;
    padding: 28px;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    background: #F0FBF9;
}}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
}}
[data-testid="stMetricLabel"] p {{
    font-weight: 500;
    color: var(--slate);
}}
[data-testid="stMetricValue"] {{
    font-family: 'Outfit', sans-serif;
    color: var(--ink);
    font-weight: 700;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    border-radius: 14px;
    padding: 0.7rem 2rem;
    border: none;
    color: #FFFFFF;
    background: linear-gradient(135deg, var(--aqua) 0%, var(--aqua-dark) 100%);
    box-shadow: 0 10px 24px -10px rgba(20,184,166,0.7);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    color: #FFFFFF;
    box-shadow: 0 14px 28px -10px rgba(20,184,166,0.85);
}}
.stButton > button:focus-visible {{
    outline: 3px solid var(--sun);
    outline-offset: 2px;
}}

/* ---------- Inputs ---------- */
[data-baseweb="select"] > div {{
    border-radius: 12px;
    border-color: var(--line);
    background: #FFFFFF;
}}

/* ---------- Dataframes & expanders ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
}}
[data-testid="stExpander"] {{
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
}}

/* ---------- Best result card ---------- */
.best-card {{
    background: linear-gradient(135deg, var(--ink) 0%, #1B2F66 100%);
    border-radius: 24px;
    padding: 32px 36px;
    margin: 8px 0 18px 0;
    box-shadow: 0 20px 48px -24px rgba(15,27,61,0.7);
}}
.best-badge {{
    display: inline-block;
    background: var(--sun);
    color: var(--ink);
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
}}
.best-card .model {{
    font-family: 'Outfit', sans-serif;
    color: #FFFFFF;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 6px;
}}
.best-card .strategy {{
    color: #9FE8DE;
    font-size: 1.1rem;
    font-weight: 500;
    margin-bottom: 14px;
}}
.best-card .config {{
    color: #C6D0E8;
    font-size: 0.95rem;
    line-height: 1.6;
    border-top: 1px solid rgba(255,255,255,0.15);
    padding-top: 14px;
}}

@media (max-width: 800px) {{
    .hero {{ padding: 40px 26px; }}
    .hero h1 {{ font-size: 2.4rem; }}
    .steps {{ grid-template-columns: 1fr 1fr; }}
}}

@media (prefers-reduced-motion: reduce) {{
    .stButton > button {{ transition: none; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


def section(title: str) -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero + how it works
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
    <h1>Preprocess<span class="iq">IQ</span></h1>
    <div class="tagline">Find the preprocessing that your model actually likes.</div>
    <p>
        Upload a tabular CSV. PreprocessIQ checks the data, fixes common
        formatting problems, tries different preprocessing strategies, and
        ranks the models that fit your task.
    </p>
</div>

<div class="steps">
    <div class="step">
        <div class="icon">📂</div>
        <h4>Upload</h4>
        <p>Drop in any tabular CSV file.</p>
    </div>
    <div class="step">
        <div class="icon">🧹</div>
        <h4>Clean</h4>
        <p>Types, missing values and duplicates are checked for you.</p>
    </div>
    <div class="step">
        <div class="icon">🎯</div>
        <h4>Pick a target</h4>
        <p>Choose what to predict and the task type.</p>
    </div>
    <div class="step">
        <div class="icon">🏆</div>
        <h4>Compare</h4>
        <p>See the best model and preprocessing combination.</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

section("Upload your dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Main flow (logic unchanged)
# ---------------------------------------------------------------------------

if uploaded_file is not None:

    try:
        raw_data = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully!")

        section("Original dataset preview")
        st.dataframe(
            raw_data.head(10),
            use_container_width=True,
        )

        with st.spinner("Inspecting and cleaning your dataset..."):
            data, report = prepare_dataset(raw_data)

        section("Cleaned dataset preview")
        st.dataframe(
            data.head(10),
            use_container_width=True,
        )

        section("Dataset overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", report["rows"])
        col2.metric("Columns", report["columns"])
        col3.metric("Missing Values", report["missing_values"])
        col4.metric("Duplicate Rows", report["duplicate_rows"])

        with st.expander("View detected column types"):
            st.write(
                "Numerical columns:",
                report["numerical_columns"],
            )
            st.write(
                "Categorical columns:",
                report["categorical_columns"],
            )
            st.write(
                "Date/time columns:",
                report["datetime_columns"],
            )

        section("Set up the benchmark")

        setup_left, setup_right = st.columns([3, 2])

        with setup_left:
            target_column = st.selectbox(
                "Select the target column to predict",
                options=data.columns.tolist(),
            )

        with setup_right:
            task_type = st.radio(
                "Select prediction task",
                options=["classification", "regression"],
                horizontal=True,
            )

        st.caption(
            "Choose classification for category predictions, "
            "or regression for numeric-value predictions."
        )

        if st.button("Run Benchmark", type="primary"):

            try:
                X, y = split_features_target(
                    data,
                    target_column,
                )

                st.info(
                    f"Benchmarking {len(X)} rows and "
                    f"{X.shape[1]} feature columns."
                )

                with st.spinner(
                    "Running model and preprocessing experiments..."
                ):
                    results = run_benchmark(
                        X,
                        y,
                        task_type=task_type,
                    )

                metric_col = (
                    "f1_score"
                    if task_type == "classification"
                    else "r2_score"
                )

                display_results = results.copy()

                if metric_col in display_results.columns:
                    display_results = display_results.sort_values(
                        by=metric_col,
                        ascending=False,
                        na_position="last",
                    )

                float_columns = display_results.select_dtypes(
                    include="float"
                ).columns
                display_results[float_columns] = display_results[
                    float_columns
                ].round(4)

                section("Benchmark results")
                st.caption("Sorted from best to worst.")
                st.dataframe(
                    display_results,
                    use_container_width=True,
                )

                successful_results = results[
                    results["status"] == "Success"
                ]

                st.write(
                    f"Successful experiments: "
                    f"{len(successful_results)} of {len(results)}"
                )

                if not successful_results.empty:

                    best = find_best_combination(
                        results,
                        task_type=task_type,
                    )

                    section("Best combination")

                    st.markdown(
                        f"""
<div class="best-card">
    <div class="best-badge">Top performer</div>
    <div class="model">{html.escape(str(best["model"]))}</div>
    <div class="strategy">with {html.escape(str(best["scaling"]))}</div>
    <div class="config"><b>Full configuration:</b> {html.escape(str(best["preprocessing"]))}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    if task_type == "classification":
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(
                            "F1-score",
                            f"{best['f1_score']:.4f}",
                        )
                        m2.metric(
                            "Accuracy",
                            f"{float(best['accuracy']):.4f}",
                        )
                        m3.metric(
                            "Precision",
                            f"{float(best['precision']):.4f}",
                        )
                        m4.metric(
                            "Recall",
                            f"{float(best['recall']):.4f}",
                        )
                    else:
                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "R² score",
                            f"{float(best['r2_score']):.4f}",
                        )
                        m2.metric(
                            "MAE",
                            f"{float(best['mae']):.2f}",
                        )
                        m3.metric(
                            "RMSE",
                            f"{float(best['rmse']):.2f}",
                        )

                    st.warning(
                        "This is a comparison on your dataset, "
                        "not a guarantee of performance on unseen data."
                    )

                else:
                    st.error(
                        "All benchmark experiments failed. "
                        "Review the error column in the results table."
                    )

            except Exception as error:
                st.error(
                    f"Benchmark could not be completed: {error}"
                )

    except Exception as error:
        st.error(
            f"Could not read or prepare the dataset: {error}"
        )