import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from PIL import Image


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="KISAN-MITRA AI",
    page_icon="🌾",
    layout="wide"
)


# =========================================================
# CROP RECOMMENDATION ENGINE
# =========================================================

CROP_CONDITIONS = {
    "Rice": [90, 40, 40, 25, 80, 6.5, 200],
    "Wheat": [70, 40, 40, 20, 60, 6.5, 100],
    "Maize": [80, 50, 50, 25, 65, 6.5, 120],
    "Cotton": [60, 40, 40, 28, 60, 7.0, 100],
    "Sugarcane": [100, 50, 50, 27, 75, 6.5, 180],
    "Groundnut": [50, 30, 40, 27, 60, 6.5, 80],
    "Millet": [40, 30, 30, 28, 60, 6.5, 70],
    "Chickpea": [40, 30, 30, 20, 50, 6.5, 60]
}


# Load the trained crop prediction model.
MODEL_PATH = Path(__file__).resolve().parent / "crop_model.pkl"

try:
    crop_model = joblib.load(MODEL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as error:
    crop_model = None
    MODEL_LOAD_ERROR = str(error)


def recommend_crops(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    if crop_model is None:
        raise RuntimeError(
            f"Could not load the trained crop model: {MODEL_LOAD_ERROR}"
        )

    farm_conditions = pd.DataFrame(
        [[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    probabilities = crop_model.predict_proba(farm_conditions)[0]

    display_names = {
        "blackgram": "Black Gram",
        "kidneybeans": "Kidney Beans",
        "mungbean": "Mung Bean",
        "mothbeans": "Moth Beans",
        "pigeonpeas": "Pigeon Peas"
    }

    results = []

    for crop, probability in zip(crop_model.classes_, probabilities):
        if probability <= 0:
            continue

        crop_name = str(crop).lower()
        display_name = display_names.get(
            crop_name,
            crop_name.title()
        )

        results.append({
            "Crop": display_name,
            "Suitability": float(probability) * 100
        })

    results.sort(
        key=lambda item: item["Suitability"],
        reverse=True
    )

    return results

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌾 KISAN-MITRA AI")

st.sidebar.caption(
    "Smart Agriculture Decision Support System"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🌱 Crop Recommendation",
        "💧 Irrigation Advisor",
        "🦠 Disease Analysis",
        "📊 Farm Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "KISAN-MITRA helps farmers with "
    "crop selection, irrigation guidance "
    "and farm monitoring."
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🌾 KISAN-MITRA AI")

    st.subheader(
        "AI-Powered Smart Agriculture Assistant"
    )

    st.write(
        "An intelligent agriculture decision-support "
        "system for smarter farming decisions."
    )

    st.divider()

    st.subheader("🌱 Farm Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌱 Crop Health",
            "84%"
        )

    with col2:
        st.metric(
            "💧 Soil Moisture",
            "48%"
        )

    with col3:
        st.metric(
            "🌡 Temperature",
            "29°C"
        )

    with col4:
        st.metric(
            "🐛 Pest Risk",
            "Medium"
        )

    st.divider()

    st.subheader("🤖 AI Farm Recommendation")

    st.success(
        "Current farm conditions are suitable for crop growth. "
        "Maintain moderate irrigation and regularly monitor "
        "the crop for pests and diseases."
    )

    st.subheader("📋 Farm Status")

    farm_data = pd.DataFrame({
        "Parameter": [
            "Crop Health",
            "Soil Moisture",
            "Temperature",
            "Pest Risk",
            "Irrigation"
        ],
        "Status": [
            "Good",
            "Moderate",
            "Normal",
            "Medium",
            "Recommended"
        ]
    })

    st.dataframe(
        farm_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🔄 KISAN-MITRA Workflow")

    st.code(
        """
Soil & Weather Data
        ↓
AI Decision Engine
        ↓
Crop Recommendation
        ↓
Irrigation Guidance
        ↓
Disease Monitoring
        ↓
Farm Analytics
        """
    )


# =========================================================
# CROP RECOMMENDATION
# =========================================================

elif page == "🌱 Crop Recommendation":

    st.header("🌱 AI Crop Recommendation")

    st.write(
        "Enter the soil and environmental conditions "
        "of your field."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=150.0,
            value=70.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=150.0,
            value=40.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=150.0,
            value=40.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=28.0
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=150.0
        )

    st.divider()

    if st.button(
        "🔍 Recommend Best Crops",
        use_container_width=True
    ):

        results = recommend_crops(
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        )

        st.subheader("🤖 AI Crop Recommendations")

        top_three = results[:3]

        for index, result in enumerate(top_three):

            crop = result["Crop"]
            score = result["Suitability"]

            if index == 0:
                label = "🥇 Best Match"
            elif index == 1:
                label = "🥈 Second Best"
            else:
                label = "🥉 Third Best"

            col1, col2 = st.columns([3, 1])

            with col1:
                st.success(
                    f"{label}: {crop}"
                )

            with col2:
                st.metric(
                "Model probability",
                    f"{score:.1f}%"
                )

        st.divider()

        st.subheader("📊 Crop Suitability Comparison")

        chart_data = pd.DataFrame(top_three)

        st.bar_chart(
            chart_data.set_index("Crop")["Suitability"]
        )

        st.info(
    "These percentages represent the model's predicted probabilities "
    "for each crop based on the entered soil and weather conditions. "
    "They are estimates from the training dataset, not guarantees of "
    "real-world crop suitability. Consider local climate, season, "
    "water availability, and advice from agricultural experts."
)


# =========================================================
# IRRIGATION ADVISOR
# =========================================================

elif page == "💧 Irrigation Advisor":

    st.header("💧 AI Irrigation Advisor")

    st.write(
        "Estimate irrigation requirements using "
        "soil moisture, temperature, humidity and rainfall."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        soil_moisture = st.slider(
            "Soil Moisture (%)",
            min_value=0,
            max_value=100,
            value=45
        )

        temperature = st.slider(
            "Temperature (°C)",
            min_value=10,
            max_value=50,
            value=30
        )

    with col2:

        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=60
        )

        rainfall_probability = st.slider(
            "Rainfall Probability (%)",
            min_value=0,
            max_value=100,
            value=20
        )

    st.divider()

    if (
        soil_moisture < 30
        and rainfall_probability < 50
    ):

        status = "🚨 Irrigation Required"

        advice = (
            "Soil moisture is low and significant rainfall "
            "is not expected. Irrigation is recommended."
        )

    elif soil_moisture < 50:

        status = "💧 Moderate Irrigation"

        advice = (
            "Soil moisture is moderate. "
            "Monitor the field and provide irrigation "
            "if the moisture level decreases."
        )

    else:

        status = "✅ No Irrigation Needed"

        advice = (
            "Soil moisture is currently sufficient. "
            "Irrigation can be postponed."
        )

    st.subheader(status)

    st.info(advice)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Soil Moisture",
            f"{soil_moisture}%"
        )

    with col2:
        st.metric(
            "Temperature",
            f"{temperature}°C"
        )

    with col3:
        st.metric(
            "Rain Probability",
            f"{rainfall_probability}%"
        )


# =========================================================
# DISEASE ANALYSIS
# =========================================================

elif page == "🦠 Disease Analysis":

    st.header("🦠 Crop Disease Analysis")

    st.write(
        "Upload a crop leaf image for the disease-analysis module."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Crop Leaf",
            use_container_width=True
        )

        st.success(
            "✅ Leaf image uploaded successfully."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Image Width",
                f"{image.width} px"
            )

        with col2:
            st.metric(
                "Image Height",
                f"{image.height} px"
            )

        
        if st.button(
            "🔍 Analyze Leaf",
            use_container_width=True
        ):

            try:
                model_path = (
                    Path(__file__).resolve().parent
                    / "disease_model.pkl"
                )

                bundle = joblib.load(model_path)
                model = bundle["model"]
                image_size = tuple(bundle["image_size"])

                # Use the same preprocessing as model training.
                processed_image = (
                    image.convert("RGB").resize(image_size)
                )

                features = (
                    np.asarray(processed_image, dtype=np.float32)
                    / 255.0
                ).reshape(1, -1)

                predicted_class = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                ranked_predictions = sorted(
                    zip(model.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                )

                st.subheader("🧪 Experimental Model Result")

                st.write(
                    "Predicted class: "
                    + str(predicted_class).replace("_", " ")
                )

                st.write("Top three model predictions:")

                for class_name, probability in ranked_predictions[:3]:
                    st.write(
                        f"- {str(class_name).replace('_', ' ')}: "
                        f"{probability:.1%}"
                    )

                st.warning(
                    "Experimental only: this model achieved 43.33% "
                    "accuracy on a 60-image test set. Its predictions "
                    "are not reliable disease diagnoses and should "
                    "not be used to decide crop treatment."
                )

            except Exception as error:
                st.error(f"Could not run the disease model: {error}")

    else:

        st.info(
            "Upload a clear crop leaf image to test this module."
        )


# =========================================================
# FARM ANALYTICS
# =========================================================

elif page == "📊 Farm Analytics":

    st.header("📊 Farm Analytics")

    st.write(
        "Monitor crop health and soil moisture trends."
    )

    days = np.arange(1, 31)

    crop_health = np.array([
        72, 74, 75, 74, 77,
        78, 80, 79, 81, 82,
        81, 83, 84, 83, 85,
        86, 85, 87, 88, 87,
        89, 88, 90, 91, 90,
        92, 91, 93, 94, 95
    ])

    soil_moisture = np.array([
        55, 53, 51, 50, 52,
        49, 48, 47, 50, 52,
        51, 49, 48, 46, 47,
        50, 52, 51, 49, 48,
        46, 47, 50, 52, 54,
        53, 55, 56, 57, 58
    ])

    st.subheader("🌱 Crop Health Trend")

    fig1, ax1 = plt.subplots()

    ax1.plot(
        days,
        crop_health,
        marker="o"
    )

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Crop Health (%)")
    ax1.set_title("Crop Health Trend")
    ax1.grid(True, alpha=0.3)

    st.pyplot(fig1)

    st.subheader("💧 Soil Moisture Trend")

    fig2, ax2 = plt.subplots()

    ax2.plot(
        days,
        soil_moisture,
        marker="o"
    )

    ax2.set_xlabel("Day")
    ax2.set_ylabel("Soil Moisture (%)")
    ax2.set_title("Soil Moisture Trend")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

    analytics_data = pd.DataFrame({
        "Day": days,
        "Crop Health (%)": crop_health,
        "Soil Moisture (%)": soil_moisture
    })

    st.subheader("📋 Farm Data")

    st.dataframe(
        analytics_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "KISAN-MITRA AI | Smart Agriculture Decision Support System"
)