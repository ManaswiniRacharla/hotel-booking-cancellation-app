import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained models
# -------------------------------
dt_model = joblib.load("hotel_booking_dt_model.pkl")
knn_model = joblib.load("hotel_booking_knn_model.pkl")
nb_model = joblib.load("hotel_booking_nb_model.pkl")

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="centered"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f0f4f8;
    font-family: 'Arial', sans-serif;
}

.title-box {
    background-color: #2c3e50;
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
}

.input-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.predict-btn {
    background-color: #27ae60;
    color: white;
    padding: 12px;
    border-radius: 8px;
    font-size: 16px;
    width: 100%;
    border: none;
    cursor: pointer;
}

.predict-btn:hover {
    background-color: #2ecc71;
}

.result-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-top: 20px;
    text-align: center;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown('<div class="title-box"><h1>🏨 Hotel Booking Cancellation Prediction</h1><p>Enter booking details to predict cancellation</p></div>', unsafe_allow_html=True)

# -------------------------------
# Month Mapping
# -------------------------------
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# -------------------------------
# Input Card
# -------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)

lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=50)
arrival_date_month = st.selectbox("Arrival Month", list(month_map.keys()))
stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=14, value=1)
stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=2)
adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
is_repeated_guest = st.selectbox("Repeated Guest?", [0,1])
previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=50, value=0)
previous_bookings_not_canceled = st.number_input("Previous Bookings Not Cancelled", min_value=0, max_value=50, value=0)
booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=400, value=0)
adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0, value=100.0)
required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=5, value=0)
total_of_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=10, value=0)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Convert month to numeric
# -------------------------------
arrival_date_month_num = month_map[arrival_date_month]

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_df = pd.DataFrame([{
    "lead_time": lead_time,
    "arrival_date_month": arrival_date_month_num,
    "stays_in_weekend_nights": stays_in_weekend_nights,
    "stays_in_week_nights": stays_in_week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "is_repeated_guest": is_repeated_guest,
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": previous_bookings_not_canceled,
    "booking_changes": booking_changes,
    "days_in_waiting_list": days_in_waiting_list,
    "adr": adr,
    "required_car_parking_spaces": required_car_parking_spaces,
    "total_of_special_requests": total_of_special_requests
}])

# -------------------------------
# Model Selection
# -------------------------------
model_choice = st.selectbox(
    "Choose Prediction Model",
    ("Decision Tree", "KNN", "Naive Bayes")
)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):
    if model_choice == "Decision Tree":
        prediction = dt_model.predict(input_df)[0]
    elif model_choice == "KNN":
        prediction = knn_model.predict(input_df)[0]
    else:
        prediction = nb_model.predict(input_df)[0]

    # -------------------------------
    # Display Result Card
    # -------------------------------
    if prediction == 1:
        st.markdown('<div class="result-card" style="color:red;">❌ Booking is likely to be CANCELLED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card" style="color:green;">✅ Booking is likely NOT CANCELLED</div>', unsafe_allow_html=True)
