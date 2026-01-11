import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

LOG_PATH = os.path.join("logs", "monitoring_logs.csv")

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
st.title("📊 Model Monitoring Dashboard (UI + Model Versions)")

# ---------- Load logs ----------
if not os.path.exists(LOG_PATH):
    st.error("No logs/monitoring_logs.csv found. Please generate logs from the prediction app first.")
    st.stop()

df = pd.read_csv(LOG_PATH)

# ---------- Validate columns ----------
required_cols = {
    "timestamp",
    "app_version",
    "model_version",
    "latency_ms",
    "prediction",
    "probability",
    "feedback_score",
    "feedback_comment",
}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Log file is missing columns: {missing}")
    st.info("Tip: Ensure log_utils.py writes the new column 'app_version' and regenerate logs.")
    st.stop()

# ---------- Convert types ----------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
df["feedback_score"] = pd.to_numeric(df["feedback_score"], errors="coerce")
df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

# ---------- Filters ----------
st.subheader("Filters")

c1, c2 = st.columns(2)

with c1:
    app_versions = ["All"] + sorted(df["app_version"].dropna().unique().tolist())
    selected_app = st.selectbox("Select App Version (UI)", app_versions)

with c2:
    model_versions = ["All"] + sorted(df["model_version"].dropna().unique().tolist())
    selected_model = st.selectbox("Select Model Version", model_versions)

filtered = df.copy()
if selected_app != "All":
    filtered = filtered[filtered["app_version"] == selected_app]
if selected_model != "All":
    filtered = filtered[filtered["model_version"] == selected_model]

st.divider()

# ---------- Summary metrics ----------
st.subheader("Summary Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Rows", int(len(filtered)))

with m2:
    st.metric("Avg Latency (ms)", round(filtered["latency_ms"].mean(), 2) if len(filtered) else 0)

with m3:
    st.metric("Avg Feedback Score", round(filtered["feedback_score"].mean(), 2) if len(filtered) else 0)

with m4:
    st.metric("Avg Probability", round(filtered["probability"].mean(), 3) if len(filtered) else 0)

st.divider()

# ---------- Grouped comparison ----------
st.subheader("Model Comparison (Grouped Summary)")

grouped = filtered.groupby(["app_version", "model_version"]).agg(
    avg_latency_ms=("latency_ms", "mean"),
    avg_feedback=("feedback_score", "mean"),
    avg_probability=("probability", "mean"),
    count=("model_version", "count"),
).reset_index()

st.write("**Per app_version + model_version summary**")
st.dataframe(grouped, use_container_width=True)

# ---------- Visualization helpers ----------
def plot_bar(x, y, xlabel, ylabel, title):
    fig = plt.figure()
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    st.pyplot(fig)

st.divider()

# ---------- Charts ----------
st.subheader("Visualizations")

# If user filtered to a single app_version, show V1 vs V2 charts clearly
if selected_app != "All" and len(grouped) > 0:
    sub = grouped[grouped["app_version"] == selected_app].sort_values("model_version")

    if len(sub) > 0:
        st.write(f"### Average Latency by Model ({selected_app})")
        plot_bar(
            sub["model_version"],
            sub["avg_latency_ms"],
            "Model Version",
            "Average Latency (ms)",
            f"Avg Latency - {selected_app}",
        )

        st.write(f"### Average Feedback Score by Model ({selected_app})")
        plot_bar(
            sub["model_version"],
            sub["avg_feedback"],
            "Model Version",
            "Average Feedback Score",
            f"Avg Feedback - {selected_app}",
        )
else:
    # General chart across model_version (ignoring app_version)
    overall = filtered.groupby("model_version").agg(
        avg_latency_ms=("latency_ms", "mean"),
        avg_feedback=("feedback_score", "mean"),
        count=("model_version", "count"),
    ).reset_index().sort_values("model_version")

    if len(overall) > 0:
        st.write("### Average Latency by Model Version (All / Filtered)")
        plot_bar(
            overall["model_version"],
            overall["avg_latency_ms"],
            "Model Version",
            "Average Latency (ms)",
            "Avg Latency by Model",
        )

        st.write("### Average Feedback Score by Model Version (All / Filtered)")
        plot_bar(
            overall["model_version"],
            overall["avg_feedback"],
            "Model Version",
            "Average Feedback Score",
            "Avg Feedback by Model",
        )

st.divider()

# ---------- Recent comments ----------
st.subheader("Recent User Comments")

comments_df = filtered[
    filtered["feedback_comment"].notna() & (filtered["feedback_comment"].astype(str).str.strip() != "")
].sort_values("timestamp", ascending=False).head(10)

if len(comments_df) == 0:
    st.info("No comments found yet in the selected filters.")
else:
    st.dataframe(
        comments_df[["timestamp", "app_version", "model_version", "feedback_score", "feedback_comment"]],
        use_container_width=True,
    )

st.divider()

# ---------- Raw logs ----------
st.subheader("Raw Monitoring Logs")
st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)

st.divider()

# ---------- Interpretation ----------
st.subheader("Interpretation (Quick)")

if len(filtered) == 0:
    st.write("- No rows available under the current filters. Try selecting 'All'.")
else:
    # Compare v1 vs v2 latency if both exist
    if set(filtered["model_version"].unique()) >= {"v1", "v2"}:
        v1_latency = filtered[filtered["model_version"] == "v1"]["latency_ms"].mean()
        v2_latency = filtered[filtered["model_version"] == "v2"]["latency_ms"].mean()

        st.write(
            f"- **Latency:** Model V2 is typically {'slower' if v2_latency > v1_latency else 'faster'} than Model V1 "
            f"(avg V1={v1_latency:.2f}ms vs V2={v2_latency:.2f}ms)."
        )

    # UI iteration narrative if both app versions exist
    if "v1_ui" in df["app_version"].unique() and "v2_ui" in df["app_version"].unique():
        st.write(
            "- **UI Iteration:** App version `v1_ui` represents the older UX (feedback before prediction). "
            "App version `v2_ui` represents the improved UX (prediction first, then feedback)."
        )
        st.write(
            "- **Monitoring use:** Compare feedback/comments between `v1_ui` and `v2_ui` to show improvement."
        )

    st.write("- **Next iteration idea:** reduce V2 latency (tune model complexity) and collect more user feedback samples.")
