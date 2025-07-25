import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# ---------- Travel Time API ----------
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjY2MTU2NWY5N2M2YTQyNDI4MWE5ODg1NTdhYmQ2NzBmIiwiaCI6Im11cm11cjY0In0="

def get_travel_time_minutes(origin, destination):
    try:
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_API_KEY}
        geocode = "https://api.openrouteservice.org/geocode/search"

        # Convert origin and destination to coordinates
        coord_o = requests.get(geocode, params={"api_key": ORS_API_KEY, "text": origin}).json()
        coord_d = requests.get(geocode, params={"api_key": ORS_API_KEY, "text": destination}).json()

        loc_o = coord_o["features"][0]["geometry"]["coordinates"]
        loc_d = coord_d["features"][0]["geometry"]["coordinates"]

        # Request route
        body = {"coordinates": [loc_o, loc_d]}
        route = requests.post(url, json=body, headers=headers).json()
        duration_sec = route["features"][0]["properties"]["summary"]["duration"]
        return round(duration_sec / 60, 1)  # minutes
    except Exception as e:
        print("Error fetching travel time:", e)
        return None

# ---------- Load model and data ----------
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

df = pd.read_csv('restaurant_data.csv')
review_texts = df['review'].tolist()
restaurant_names = df['name'].tolist()
review_embeddings = model.encode(review_texts, normalize_embeddings=True)

# ---------- UI ----------
st.title("🍽️ DateDine AI: Maximize Your Date Time")
st.markdown("Find restaurants that fit both your vibe **and your schedule**.")

# 🎯 Vibe Inputs
vibe1 = st.text_input("User 1: What's your ideal vibe?", "Cozy, candlelight, quiet")
vibe2 = st.text_input("User 2: What's your ideal vibe?", "Rooftop, romantic, good coffee")

# 🕒 Location + Time Inputs
st.markdown("---")
st.subheader("🕒 Time Constraints & Location")

col1, col2 = st.columns(2)
with col1:
    loc1 = st.text_input("User 1: Starting location", "Powai, Mumbai")
    time1 = st.number_input("User 1: Total time (minutes)", min_value=30, max_value=300, value=120, step=10)
with col2:
    loc2 = st.text_input("User 2: Starting location", "Dadar, Mumbai")
    time2 = st.number_input("User 2: Total time (minutes)", min_value=30, max_value=300, value=120, step=10)

# ---------- Button + Logic ----------
if st.button("Find Restaurant"):

    # Embed user preferences
    user_vecs = model.encode([vibe1, vibe2], normalize_embeddings=True)
    avg_pref = np.mean(user_vecs, axis=0).reshape(1, -1)

    # Search using FAISS
    index = faiss.IndexFlatIP(review_embeddings.shape[1])
    index.add(review_embeddings)
    D, I = index.search(avg_pref, k=5)  # Top 5 vibe matches

    recommendations = []
    for idx in I[0]:
        restaurant = df.iloc[idx]
        r_name = restaurant['name']
        r_location = restaurant.get('location', f"{r_name}, Mumbai")

        # Get round-trip travel times
        travel1 = get_travel_time_minutes(loc1, r_location)
        travel2 = get_travel_time_minutes(loc2, r_location)

        if travel1 is None or travel2 is None:
            continue

        total_rt1 = 2 * travel1
        total_rt2 = 2 * travel2

        # Calculate effective dating time
        max_dating_time1 = time1 - total_rt1
        max_dating_time2 = time2 - total_rt2
        effective_dating_time = min(max_dating_time1, max_dating_time2)

        if effective_dating_time >= 30:
            recommendations.append({
                'name': r_name,
                'review': restaurant['review'],
                'rt1': total_rt1,
                'rt2': total_rt2,
                'date_time': effective_dating_time
            })

    # Sort and display
    recommendations.sort(key=lambda x: x['date_time'], reverse=True)

    st.subheader("⏱️ Smart Time-Optimized Picks:")
    if not recommendations:
        st.warning("😢 No restaurant found that fits both vibes and time limits.")
    for r in recommendations[:3]:
        st.markdown(f"**{r['name']}**")
        st.write(r['review'])
        st.write(f"💑 Max dating time: {round(r['date_time'])} min")
        st.write(f"🚗 User 1 travel: {round(r['rt1'])} min | User 2 travel: {round(r['rt2'])} min")
        st.markdown("---")
