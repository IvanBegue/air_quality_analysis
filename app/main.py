import html
import pickle
from numbers import Integral, Real
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path



st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown(
    """
    <style>
        :root {
            --navy: #10264b;
            --muted: #53627a;
            --border: #dbe4ef;
            --surface: #ffffff;
            --background: #f6f9fc;
            --primary: #2f80ed;
            --green: #20a447;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, #ffffff 0%, #f7faff 45%, #f4f8fc 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%);
            border-right: 1px solid #dce7f3;
            min-width: 330px;
            max-width: 330px;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.6rem;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }

        h1, h2, h3, p, label {
            color: var(--navy);
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 0.5rem;
        }

        .sidebar-brand-icon {
            font-size: 3.1rem;
            line-height: 1;
        }

        .sidebar-brand-title {
            color: var(--navy);
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .sidebar-description {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.5;
            margin-bottom: 1.2rem;
        }

        .page-title {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            color: var(--navy);
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
        }

        .page-title-icon {
            font-size: 2.8rem;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin-bottom: 1.35rem;
        }

        .prediction-card {
            display: flex;
            align-items: center;
            gap: 1.25rem;
            background: linear-gradient(110deg, #f1fff5 0%, #f8fffa 58%, #ffffff 100%);
            border: 1px solid #b8e8c6;
            border-radius: 14px;
            padding: 1.4rem 1.55rem;
            margin: 0.4rem 0 1.25rem 0;
            box-shadow: 0 8px 28px rgba(22, 156, 70, 0.06);
        }

        .prediction-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 86px;
            height: 86px;
            min-width: 86px;
            border-radius: 50%;
            color: white;
            font-size: 2.55rem;
            background: linear-gradient(145deg, #31bd5b, #128c39);
            box-shadow: 0 10px 24px rgba(22, 156, 70, 0.22);
        }

        .prediction-label {
            color: var(--navy);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .prediction-value {
            color: #179845;
            font-size: 2.15rem;
            font-weight: 850;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }

        .confidence-line {
            color: var(--navy);
            font-weight: 650;
        }

        .confidence-badge {
            display: inline-block;
            margin-left: 0.35rem;
            background: #1f9d49;
            color: white;
            font-weight: 750;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            box-shadow: 0 3px 10px rgba(31, 157, 73, 0.16);
        }

        .empty-card {
            background: linear-gradient(110deg, #f7fbff, #ffffff);
            border: 1px dashed #b8cbe0;
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            margin: 0.4rem 0 1.25rem 0;
            color: var(--muted);
        }

        .section-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 8px 26px rgba(18, 46, 84, 0.045);
            min-height: 100%;
        }

        .section-title {
            color: var(--navy);
            font-size: 1.13rem;
            font-weight: 800;
            margin-bottom: 0.85rem;
        }

        .probability-table {
            width: 100%;
            background: #ffffff;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--navy);
            font-size: 0.93rem;
        }

        .probability-table th {
            background: #f7f9fc;
            padding: 0.78rem 0.85rem;
            text-align: left;
            font-weight: 800;
            border-bottom: 1px solid var(--border);
        }

        .probability-table th:last-child,
        .probability-table td:last-child {
            text-align: right;
        }

        .probability-table td {
            background: #ffffff;
            padding: 0.78rem 0.85rem;
            border-bottom: 1px solid #e8eef5;
        }

        .probability-table tr:last-child td {
            border-bottom: none;
        }

        .probability-dot {
            display: inline-block;
            width: 11px;
            height: 11px;
            border-radius: 50%;
            margin-right: 0.65rem;
            vertical-align: middle;
        }

        .probability-note {
            margin-top: 0.9rem;
            padding: 0.8rem 0.95rem;
            background: #f8fbff;
            border: 1px solid #dce8f4;
            border-radius: 10px;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }


        /* Keep the Altair probability chart on a white surface */
        [data-testid="stVegaLiteChart"] {
            background: #ffffff !important;
            border-radius: 10px;
            padding: 0.35rem;
        }

        [data-testid="stVegaLiteChart"] > div {
            background: #ffffff !important;
        }

        .metrics-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.15rem 1.2rem 1.25rem;
            margin-top: 1.2rem;
            box-shadow: 0 8px 26px rgba(18, 46, 84, 0.045);
        }

        .metric-card {
            height: 118px;
            background: linear-gradient(145deg, #ffffff, #f8fbff);
            border: 1px solid #dce5ef;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 5px 16px rgba(18, 46, 84, 0.045);
        }

        .metric-heading {
            color: var(--navy);
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: #0d2144;
            font-size: 1.65rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .metric-subtext {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }

        .about-card {
            margin-top: 1.25rem;
            padding: 0.95rem 1rem;
            border-radius: 11px;
            border: 1px solid #bfdbf5;
            background: linear-gradient(145deg, #f5faff, #edf6ff);
            color: var(--navy);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .about-title {
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .footer {
            text-align: center;
            color: #65758a;
            font-size: 0.86rem;
            margin-top: 1.3rem;
            padding-bottom: 0.2rem;
        }

        div.stButton > button {
            width: 100%;
            min-height: 46px;
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 750;
            background: linear-gradient(180deg, #3c91f6, #216dcc);
            box-shadow: 0 7px 17px rgba(47, 128, 237, 0.22);
        }

        div.stButton > button:hover {
            color: white;
            border: none;
            transform: translateY(-1px);
            background: linear-gradient(180deg, #2f80ed, #185fb9);
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            [data-testid="stSidebar"] {
                min-width: auto;
                max-width: none;
            }

            .prediction-card {
                align-items: flex-start;
            }

            .prediction-icon {
                width: 68px;
                height: 68px;
                min-width: 68px;
                font-size: 2rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)



base_path = Path(__file__).resolve().parent

model_path = (
    base_path
    / "models"
    / "random_forest_air_quality.pkl"
)

@st.cache_resource
def load_model_bundle(
    model_file: str,
    file_modified_time: float,
):
    """
    Reload the bundle automatically whenever the pickle file changes.
    """
    del file_modified_time

    model_file_path = Path(model_file)

    if not model_file_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file_path}"
        )

    with open(model_file_path, "rb") as pickle_file:
        return pickle.load(pickle_file)


try:
    model_bundle = load_model_bundle(
        str(model_path),
        model_path.stat().st_mtime,
    )
except Exception as error:
    st.error(
        f"The prediction model could not be loaded: {error}"
    )
    st.stop()


model = model_bundle["model"]
countries = model_bundle["countries"]


# ===================================================
# 4. DISPLAY SETTINGS
# ===================================================

class_order = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
]

class_colours = {
    "Good": "#20a447",
    "Moderate": "#ffc20a",
    "Unhealthy for Sensitive Groups": "#ff7a00",
    "Unhealthy": "#ef2b2d",
    "Very Unhealthy": "#9d3ec5",
}


def is_valid_number(value):
    """Return True for Python/NumPy numeric values that are not NaN."""
    return isinstance(value, Real) and not pd.isna(value)


def format_percentage(value):
    if is_valid_number(value):
        return f"{float(value):.2%}"
    return "N/A"


def format_decimal(value):
    if is_valid_number(value):
        return f"{float(value):.4f}"
    return "N/A"


def get_bundle_value(*possible_keys, default=None):
    """Read a value even when an older pickle used a different key name."""
    for key in possible_keys:
        value = model_bundle.get(key)
        if value is not None and not pd.isna(value):
            return value
    return default


def get_model_classes():
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        final_estimator = model.named_steps.get("model")
        if final_estimator is not None and hasattr(
            final_estimator,
            "classes_",
        ):
            return list(final_estimator.classes_)

    raise AttributeError(
        "The trained model does not expose class labels."
    )


# ===================================================
# 5. SIDEBAR
# ===================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🌍</div>
            <div class="sidebar-brand-title">
                Air Quality<br>Prediction
            </div>
        </div>
        <div class="sidebar-description">
            Enter the details below to predict the air-quality level.
        </div>
        """,
        unsafe_allow_html=True,
    )

    country = st.selectbox(
        "Country",
        countries,
    )

    year = st.number_input(
        "Year",
        min_value=1950,
        max_value=2100,
        value=2024,
        step=1,
    )

    pm10 = st.number_input(
        "PM10 Concentration (µg/m³)",
        min_value=0.0,
        value=35.0,
        step=1.0,
        format="%.2f",
    )

    no2 = st.number_input(
        "NO₂ Concentration (µg/m³)",
        min_value=0.0,
        value=18.0,
        step=1.0,
        format="%.2f",
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=-20.3484,
        step=0.0001,
        format="%.6f",
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=57.5522,
        step=0.0001,
        format="%.6f",
    )

    predict_clicked = st.button(
        "📈  Predict Air Quality",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="about-card">
            <div class="about-title">ℹ️ &nbsp; About</div>
            This prediction is based on a Random Forest
            machine-learning model trained on historical
            air-quality data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================================================
# 6. PREDICTION
# ===================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


if predict_clicked:
    input_data = pd.DataFrame({
        "country_name": [country],
        "year": [year],
        "pm10_concentration": [pm10],
        "no2_concentration": [no2],
        "latitude": [latitude],
        "longitude": [longitude],
    })

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        model_classes = get_model_classes()

        probability_table = pd.DataFrame({
            "Air-quality class": model_classes,
            "Probability": probabilities,
        })

        st.session_state.prediction_result = {
            "prediction": prediction,
            "probability_table": probability_table,
        }

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
        st.stop()


# ===================================================
# 7. MAIN HEADER
# ===================================================

st.markdown(
    """
    <div class="page-title">
        <span class="page-title-icon">☁️</span>
        <span>Air Quality Level Prediction</span>
    </div>
    <div class="page-subtitle">
        This application predicts the air-quality level based on
        environmental and geographical factors.
    </div>
    """,
    unsafe_allow_html=True,
)


# ===================================================
# 8. PREDICTION RESULT AREA
# ===================================================

result = st.session_state.prediction_result

if result is None:
    st.markdown(
        """
        <div class="empty-card">
            Enter the environmental information in the sidebar and
            click <strong>Predict Air Quality</strong> to display the
            predicted category and probability distribution.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    prediction = result["prediction"]
    probability_table = result["probability_table"].copy()

    highest_probability = float(
        probability_table["Probability"].max()
    )

    safe_prediction = html.escape(str(prediction))

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-icon">🌿</div>
            <div>
                <div class="prediction-label">
                    Predicted Air Quality Level
                </div>
                <div class="prediction-value">
                    {safe_prediction}
                </div>
                <div class="confidence-line">
                    Prediction Confidence:
                    <span class="confidence-badge">
                        {highest_probability:.2%}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_table["Order"] = (
        probability_table["Air-quality class"]
        .map({
            label: index
            for index, label in enumerate(class_order)
        })
        .fillna(len(class_order))
    )

    probability_table = probability_table.sort_values(
        by="Order"
    )

    probability_table["Probability (%)"] = (
        probability_table["Probability"]
        .mul(100)
        .round(2)
    )

    probability_column, chart_column = st.columns(
        [0.92, 1.45],
        gap="medium",
    )

    with probability_column:
        table_rows = []

        for _, row in probability_table.iterrows():
            class_name = str(row["Air-quality class"])
            colour = class_colours.get(
                class_name,
                "#64748b",
            )

            table_rows.append(
                "<tr>"
                "<td>"
                f'<span class="probability-dot" '
                f'style="background:{colour};"></span>'
                f"{html.escape(class_name)}"
                "</td>"
                f'<td>{row["Probability (%)"]:.2f}%</td>'
                "</tr>"
            )

        probability_table_html = (
            '<div class="section-card">'
            '<div class="section-title">'
            '📊 &nbsp; Prediction Probabilities'
            '</div>'
            '<table class="probability-table">'
            '<thead>'
            '<tr>'
            '<th>Air Quality Level</th>'
            '<th>Probability</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            + "".join(table_rows)
            + '</tbody>'
            '</table>'
            '<div class="probability-note">'
            "The probabilities above indicate the model's "
            "confidence for each air-quality category."
            '</div>'
            '</div>'
        )

        st.markdown(
            probability_table_html,
            unsafe_allow_html=True,
        )

    with chart_column:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">
                    ◕ &nbsp; Probability Distribution
                </div>
            """,
            unsafe_allow_html=True,
        )

        chart_order = [
            label
            for label in class_order
            if label in probability_table[
                "Air-quality class"
            ].tolist()
        ]

        # A horizontal chart keeps every long category label visible and
        # prevents value labels from being clipped at the top of the chart.
        probability_table["Probability label"] = (
            probability_table["Probability (%)"]
            .map(lambda value: f"{value:.2f}%")
        )

        bars = (
            alt.Chart(probability_table)
            .mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5,
                size=34,
            )
            .encode(
                y=alt.Y(
                    "Air-quality class:N",
                    title=None,
                    sort=chart_order,
                    axis=alt.Axis(
                        labelLimit=260,
                        labelPadding=10,
                        labelOverlap=False,
                    ),
                ),
                x=alt.X(
                    "Probability (%):Q",
                    title="Probability (%)",
                    scale=alt.Scale(domain=[0, 105]),
                ),
                color=alt.Color(
                    "Air-quality class:N",
                    scale=alt.Scale(
                        domain=list(class_colours.keys()),
                        range=list(class_colours.values()),
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip(
                        "Air-quality class:N",
                        title="Class",
                    ),
                    alt.Tooltip(
                        "Probability (%):Q",
                        title="Probability",
                        format=".2f",
                    ),
                ],
            )
        )

        labels = (
            alt.Chart(probability_table)
            .mark_text(
                align="left",
                baseline="middle",
                dx=7,
                fontSize=12,
                fontWeight="bold",
                color="#10264b",
            )
            .encode(
                y=alt.Y(
                    "Air-quality class:N",
                    sort=chart_order,
                ),
                x=alt.X("Probability (%):Q"),
                text=alt.Text("Probability label:N"),
            )
        )

        probability_chart = (
            bars + labels
        ).properties(
            height=330,
            padding={"left": 5, "right": 45, "top": 10, "bottom": 5},
        ).configure(
            background="#ffffff"
        ).configure_view(
            fill="#ffffff",
            stroke=None
        ).configure_axis(
            gridColor="#e8eef5",
            domainColor="#cbd5e1",
            labelColor="#10264b",
            titleColor="#10264b"
        )

        st.altair_chart(
            probability_chart,
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )




test_accuracy = get_bundle_value(
    "test_accuracy",
    "accuracy",
    "accuracy_score",
											 
)

balanced_accuracy = get_bundle_value(
    "balanced_accuracy",
    "balanced_accuracy_score",
												 
)

macro_f1 = get_bundle_value(
    "macro_f1",
    "macro_f1_score",
    "f1_macro",
										
)

weighted_f1 = get_bundle_value(
    "weighted_f1",
    "weighted_f1_score",
    "f1_weighted",
										   
)

test_samples = get_bundle_value(
    "test_samples",
    "n_test_samples",
    "test_sample_count",
											
)

total_samples = get_bundle_value(
    "total_samples",
    "n_samples",
    "sample_count",
											 
)

actual_test_fraction = get_bundle_value(
    "actual_test_fraction",
    "test_fraction",
    "test_size",
)

if isinstance(test_samples, Integral):
							  
  
    test_samples = int(test_samples)

if isinstance(total_samples, Integral):
    total_samples = int(total_samples)


if not is_valid_number(actual_test_fraction):
    if (
        isinstance(test_samples, int)
        and isinstance(total_samples, int)
        and total_samples > 0
    ):
        actual_test_fraction = (
            test_samples / total_samples
        )

if is_valid_number(actual_test_fraction):
    test_sample_subtext = (
        f"{float(actual_test_fraction):.2%} of total data"
    )
else:
    test_sample_subtext = "Testing observations"


st.markdown(
    """
    <div class="metrics-section">
        <div class="section-title">
            Model Performance (Random Forest)
        </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5, gap="small")

metric_data = [
    (
        "Accuracy",
        format_decimal(test_accuracy),
        format_percentage(test_accuracy),
    ),
    (
        "Balanced Accuracy",
        format_decimal(balanced_accuracy),
        format_percentage(balanced_accuracy),
    ),
    (
        "Macro F1-score",
        format_decimal(macro_f1),
        format_percentage(macro_f1),
    ),
    (
        "Weighted F1-score",
        format_decimal(weighted_f1),
        format_percentage(weighted_f1),
    ),
    (
        "Test Samples",
        (
            f"{test_samples:,}"
            if isinstance(test_samples, int)
            else "N/A"
        ),
        test_sample_subtext,
    ),
]

for column, (
    heading,
    value,
    subtext,
) in zip(metric_columns, metric_data):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-heading">
                    {heading}
                </div>
                <div class="metric-value">
                    {value}
                </div>
                <div class="metric-subtext">
                    {subtext}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)

													


st.markdown(
    """
    <div class="footer">
        Air Quality Prediction App
        &nbsp;|&nbsp;
        Powered by Machine Learning 
        &nbsp;|&nbsp;
        Built With ❤️
        <a
            href="https://www.linkedin.com/in/begueivan/"
            target="_blank"
            rel="noopener noreferrer"
        >
            IVAN
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
