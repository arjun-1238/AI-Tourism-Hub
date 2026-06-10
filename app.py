import streamlit as st
import pandas as pd
import pickle
import datetime
import calendar
import holidays
from groq import Groq
from dotenv import load_dotenv
import os

# Page CONFIG
st.set_page_config(
    page_title="AI Tourism Hub",
    page_icon="🏛️",
    layout="wide"
)

load_dotenv()

st.markdown("""
<style>
/* App Background */
[data-testid="stAppViewContainer"] {
    max-height: 98vh;
}

.stApp {
    background-image: linear-gradient(
        135deg, 
        rgba(241, 245, 249, 0.68) 0%, 
        rgba(226, 232, 240, 0.68) 100%
    ), 
    url("https://images.unsplash.com/photo-1548013146-72479768bada?q=80&w=1476&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: 'Segoe UI', system-ui, sans-serif;
}

/* Padding for Screen Fit */
.block-container {
    padding-top: 3.8rem !important; /* Pushes the titles comfortably below the top edge */
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* gaps between layout elements */
[data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}

/* Header Adjustments */
.centered-title {
    text-align: center;
    color: #0f172a !important;
    font-size: 32px !important; 
    font-weight: 900 !important;
    letter-spacing: -1px;
    margin-top: 0px !important;
    margin-bottom: 12px !important;
    background: linear-gradient(90deg, #1e3a8a, #2563eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Panel Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 16px; 
    border: 1px solid rgba(255, 255, 255, 0.7);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    margin-bottom: 10px;
}

/* Section Headers inside Panels */
h2 {
    color: #1e3a8a !important;
    font-size: 20px !important; 
    font-weight: 700 !important;
    border-left: 5px solid #2563eb !important;
    padding-left: 10px !important;
    margin-bottom: 15px !important;
    margin-top: 2px !important;
}

/* Tab Configurations */
button[data-baseweb="tab"] {
    font-size: 15px !important; 
    font-weight: 700 !important;
    color: #475569 !important;
    background: rgba(255, 255, 255, 0.6) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 8px 20px !important; 
    border: none !important;
    margin-right: 5px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border-bottom: 3px solid #2563eb !important;
}

/* Input Widgets */
label[data-testid="stWidgetLabel"] p {
    font-size: 14px !important; 
    font-weight: 600 !important;
    color: #1e293b !important;
}
div[data-baseweb="select"] > div, .stDateInput input {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    background: white !important;
    font-size: 14px !important;
    font-weight: 500;
}

/* Minimize space under input containers */
div[data-testid="stFormSubmitButton"], div.stSlider, div.stSelectbox {
    margin-bottom: -10px !important;
}

/* High Contrast Displays */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%) !important;
    padding: 12px 20px !important; 
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(30, 58, 138, 0.15) !important;
}
[data-testid="stMetricLabel"] {
    color: #93c5fd !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 30px !important; 
    font-weight: 800 !important;
}

/* Action Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 20px !important; 
    font-size: 15px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
}

/* Chat Area */
div[data-testid="stChatMessageContainer"] {
    background: linear-gradient(
        135deg, 
        rgba(255, 255, 255, 0.92) 0%, 
        rgba(240, 246, 255, 0.88) 100%
    ) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px !important;
    border: 2px solid rgba(37, 99, 235, 0.8) !important; 
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.08) !important;
    padding: 15px !important; 
    font-size: 15px !important;
    margin-bottom: -10px !important;
}
</style>
""", unsafe_allow_html=True)


# Model Loading
@st.cache_data
def load_data():
    return pd.read_csv("D:/planar/budget_dataset.csv")


@st.cache_resource
def load_budget_model():
    with open("D:/planar/synthetic_budget.pkl", "rb") as f:
        budget_model = pickle.load(f)
    with open("D:/planar/budget_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    return budget_model, model_columns


@st.cache_resource
def load_crowd_model():
    with open("D:/planar/crowd_predictor_model.pkl", "rb") as f:
        crowd_model = pickle.load(f)
    with open("D:/planar/monument_encoder.pkl", "rb") as f:
        monument_encoder = pickle.load(f)
    with open("D:/planar/circle_encoder.pkl", "rb") as f:
        circle_encoder = pickle.load(f)
    return crowd_model, monument_encoder, circle_encoder


try:
    df = load_data()
    budget_model, model_columns = load_budget_model()
    crowd_model, le_monument, le_circle = load_crowd_model()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.sidebar.error(f"Models missing: {e}")

# Initialize session
if "calculated_budget" not in st.session_state:
    st.session_state.calculated_budget = None
if "selected_monument" not in st.session_state:
    st.session_state.selected_monument = "Unspecified Location"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Split screen
left_column, right_column = st.columns([11, 8], gap="large")

# Left Column
with left_column:
    st.markdown('<div class="centered-title">🏛️ AI Tourism Analytics</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "💰 Budget Predictor",
        "📈 Crowd Forecast"
    ])

    # Budget Planar
    with tab1:
        st.header("💰 Travel Budget Prediction")

        b_col1, b_col2 = st.columns(2, gap="medium")
        with b_col1:
            if models_loaded:
                monument = st.selectbox("🏛️ Select Monument", sorted(df["monument"].unique()), key="b_monument")
                hotel_type = st.selectbox("🏨 Hotel Type", sorted(df["hotel_type"].unique()), key="b_hotel")
            else:
                monument = st.selectbox("🏛️ Select Monument", ["Sample Monument"], key="b_monument")
                hotel_type = st.selectbox("🏨 Hotel Type", ["Standard"], key="b_hotel")

            distance_km = st.slider("📍 Distance (km)", 1, 3000, 500, key="b_distance")

        with b_col2:
            if models_loaded:
                transport_options = [x for x in sorted(df["transport_mode"].unique()) if x.lower() != "flight"]
                transport_mode = st.selectbox("🚗 Transport Mode", transport_options, key="b_transport")
                season = st.selectbox("🌤️ Season", sorted(df["season"].unique()), key="b_season")
            else:
                transport_mode = st.selectbox("🚗 Transport Mode", ["Cab"], key="b_transport")
                season = st.selectbox("🌤️ Season", ["Summer"], key="b_season")

            travelers = st.slider("👨‍👩‍👧‍👦 Travelers", 1, 20, 2, key="b_travelers")

        st.write("")
        trip_days = st.slider("📅 Trip Days", 1, 30, 3, key="b_days")

        if st.button("✨ Calculate Estimated Budget", key="b_btn", use_container_width=True):
            if models_loaded:
                user_df = pd.DataFrame({
                    "monument": [monument], "distance_km": [distance_km], "trip_days": [trip_days],
                    "travelers": [travelers], "hotel_type": [hotel_type], "transport_mode": [transport_mode],
                    "season": [season]
                })
                user_encoded = pd.get_dummies(user_df).reindex(columns=model_columns, fill_value=0)
                prediction = budget_model.predict(user_encoded)[0]

                # Store values inside state engine
                st.session_state.calculated_budget = f"₹{prediction:,.2f}"
                st.session_state.selected_monument = monument

                st.write("")
                st.metric(label="Calculated Predictive Cost Allocation", value=st.session_state.calculated_budget)
            else:
                st.error("Budget calculations unavailable.")

    # Crowd Forecast
    with tab2:
        st.header("📈 Daily Crowd Forecast")

        c_col1, c_col2 = st.columns(2, gap="medium")
        with c_col1:
            if models_loaded:
                monument_name = st.selectbox("🏛️ Select Target Monument", le_monument.classes_, key="c_monument")
            else:
                monument_name = st.selectbox("🏛️ Select Target Monument", ["Sample"], key="c_monument")

        with c_col2:
            selected_date = st.date_input("📅 Choose Date", datetime.date.today(), key="c_date")

        if st.button("🔍 Forecast Crowd Trends", key="c_btn", use_container_width=True):
            if models_loaded:
                year, month = selected_date.year, selected_date.month
                india_holidays = holidays.India(years=year)

                monument_code = le_monument.transform([monument_name])[0]

                region_code = 0

                single_X = pd.DataFrame([[
                    month, selected_date.weekday(),
                    1 if selected_date in india_holidays else 0,
                    1 if selected_date.weekday() >= 5 else 0,
                    monument_code, region_code
                ]], columns=["month", "day_of_week", "is_holiday", "is_weekend", "monument_encoded", "circle_encoded"])

                visitors = int(crowd_model.predict(single_X)[0])

                st.write("")
                if visitors < 3000:
                    st.success(f"🟢 Low Density Forecast: {visitors:,} anticipated visitors")
                elif visitors < 5500:
                    st.warning(f"🟡 Moderate Density Forecast: {visitors:,} anticipated visitors")
                else:
                    st.error(f"🔴 Heavy Surge Density Forecast: {visitors:,} anticipated visitors")

                num_days = calendar.monthrange(year, month)[1]
                dates = pd.date_range(start=datetime.date(year, month, 1), periods=num_days)

                rows = []
                for d in dates:
                    rows.append([
                        d.month, d.weekday(),
                        1 if d in india_holidays else 0,
                        1 if d.weekday() >= 5 else 0,
                        monument_code, region_code
                    ])

                X_month = pd.DataFrame(rows,
                                       columns=["month", "day_of_week", "is_holiday", "is_weekend", "monument_encoded",
                                                "circle_encoded"])
                preds = crowd_model.predict(X_month)
                result_df = pd.DataFrame(
                    {"Date": dates, "Expected Visitors": preds.astype(int), "DayName": dates.day_name()})

                # Graphs
                st.write("")
                st.markdown("### 📊 Chart 1: Month-wide Crowd Trend Line")
                st.line_chart(result_df.set_index("Date")["Expected Visitors"])

                st.write("")
                st.markdown("### 🏛️ Chart 2: Weekly Surge Distribution (Bar Chart)")
                weekly_averages = result_df.groupby("DayName")["Expected Visitors"].mean().reindex(
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                )
                st.bar_chart(weekly_averages)

                st.write("")
                st.markdown("### 🧩 Chart 3: Passenger Surge Driver Breakdown")
                weekend_avg = result_df[result_df["Date"].dt.weekday >= 5]["Expected Visitors"].mean()
                weekday_avg = result_df[result_df["Date"].dt.weekday < 5]["Expected Visitors"].mean()

                driver_data = pd.DataFrame({
                    "Driver Factor Type": ["Weekend Density Factor", "Weekday Base Density"],
                    "Estimated Average Volume": [int(weekend_avg), int(weekday_avg)]
                })
                st.dataframe(driver_data, use_container_width=True, hide_index=True)
            else:
                st.error("Models missing.")

# Right Column
with right_column:
    st.markdown('<div class="centered-title">💬 AI Trip Assistant</div>', unsafe_allow_html=True)

    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None

    if not api_key:
        st.info("💡 Pass your GROQ_API_KEY config to execute active agent responses.")

    if st.session_state.calculated_budget:
        st.markdown(
            f'<div style="text-align:center; margin-bottom:8px; font-weight:600; color:#1e293b; font-size:16px;">🎯 Context Ready: {st.session_state.selected_monument} ({st.session_state.calculated_budget})</div>',
            unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom:15px;">', unsafe_allow_html=True)
        if st.button("🤖 Send Dashboard Data to AI", key="sync_data_btn", use_container_width=True):
            if not client:
                st.error("API client uninitialized. Check your API key configuration.")
            else:
                context_prompt = (
                    f"SYSTEM USER UPDATE: I have set my tourism dashboard to explore '{st.session_state.selected_monument}'. "
                    f"My calculated ML predictive trip budget total is {st.session_state.calculated_budget} for {st.session_state.get('b_days', 3)} days. "
                    f"Please acknowledge these parameters and suggest a custom, highly optimized daily schedule for my trip!"
                )

                st.session_state.messages.append({"role": "user", "content": context_prompt})

                try:
                    with st.spinner("AI analyzing your dashboard parameters..."):
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=st.session_state.messages
                        )
                        reply = response.choices[0].message.content

                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()

                except Exception as e:
                    st.error(f"Instant agent response processing failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat Window
    chat_container = st.container(height=400)

    with chat_container:
        for msg in st.session_state.messages:
            if "SYSTEM USER UPDATE:" in msg["content"]:
                with st.chat_message("user"):
                    st.write(f"🔄 Synced dashboard context parameters for **{st.session_state.selected_monument}**")
            else:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    if prompt := st.chat_input("Plan for your trip..."):
        if not client:
            st.error("API client uninitialized.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)

            try:
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("AI thinking..."):
                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=st.session_state.messages
                            )
                            reply = response.choices[0].message.content
                            st.write(reply)

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

            except Exception as e:
                st.error(f"API Stream error: {e}")