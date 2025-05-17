import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import pickle
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Configure page
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌾")

# Add custom CSS to style the sidebar
st.markdown("""
<style>
    .sidebar-button {
        width: 100%;
        margin: 5px 0px;
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .sidebar-button:hover {
        background-color: #f0f2f6;
    }
    div[data-testid="stSidebar"] {
        padding-top: 2rem;
    }
    .sidebar-title {
        text-align: left;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🌾 Smart Agriculture AI")

# Initialize session state for navigation
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Ask AI (Gemini)"

# Sidebar with custom navigation
with st.sidebar:
    st.markdown('<p class="sidebar-title">Main Menu</p>', unsafe_allow_html=True)
    # Create navigation buttons
    if st.button("Ask AI (Gemini)", key="btn_gemini", use_container_width=True):
        st.session_state.menu_choice = "Ask AI (Gemini)"
        st.rerun()
    if st.button("Weather Prediction", key="btn_cuaca", use_container_width=True):
        st.session_state.menu_choice = "Weather Prediction"
        st.rerun()
    if st.button("Plant Disease Detection", key="btn_penyakit", use_container_width=True):
        st.session_state.menu_choice = "Plant Disease Detection"
        st.rerun()
    if st.button("Soil Type Detection", key="btn_tanah", use_container_width=True):
        st.session_state.menu_choice = "Soil Type Detection"
        st.rerun()
    if st.button("Harvest Prediction", key="btn_panen", use_container_width=True):
        st.session_state.menu_choice = "Harvest Prediction"
        st.rerun()

# Display content based on choice
choice = st.session_state.menu_choice

# Now implement each page based on the choice
if choice == "Weather Prediction":
    st.header("🌦️ Weather Prediction By City")
    city = st.text_input("Enter City Name", "Jakarta")
    api_key = "0d402044f615b840fb0d0e167bb8b23e"  # Ganti dengan API key WeatherStack
    if st.button("Prediction"):
        url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
        response = requests.get(url).json()
        if "current" not in response:
            st.error("City not found or API key is incorrect!")
        else:
            temp = response["current"]["temperature"]
            humidity = response["current"]["humidity"]
            weather = response["current"]["weather_descriptions"][0]
            st.success(f"Weather in {city}: {weather}")
            st.write(f"🌡️ Temperature: {temp}°C")
            st.write(f"💧 Humidity: {humidity}%")

elif choice == "Ask AI (Gemini)":
    st.header("🤖 Ask AI Using Gemini")
    st.markdown("Enter any question or prompt related to farming:")
    user_prompt = st.text_area("Prompt", placeholder="Example: How do you care for chili plants to get maximum harvest results?")
    if st.button("Gemini asked"):
        if user_prompt.strip() == "":
            st.warning("Please enter a prompt first.")
        else:
            # Menampilkan indikator loading
            with st.spinner("Currently processing inquiries..."):
                # Periksa apakah user menanyakan tentang waktu atau tanggal
                waktu_keywords = ["hour", "time", "date", "today", "now", "what day", "what month", "what year", "what time"]
                pertanyaan_waktu = any(keyword in user_prompt.lower() for keyword in waktu_keywords)
                # Jika menanyakan tentang waktu, siapkan informasi waktu
                if pertanyaan_waktu:
                    import datetime
                    import pytz
                    # Gunakan timezone default Indonesia (WIB)
                    timezone_code = "Asia/Jakarta"
                    timezone_label = "WIB (Waktu Indonesia Barat)"
                    try:
                        # Dapatkan waktu saat ini berdasarkan timezone
                        tz = pytz.timezone(timezone_code)
                        current_time = datetime.datetime.now(tz)
                        # Format waktu dengan berbagai format yang mungkin diperlukan
                        waktu_lengkap = current_time.strftime("%A, %d %B %Y, %H:%M:%S %Z")
                        jam = current_time.strftime("%H:%M")
                        tanggal = current_time.strftime("%d %B %Y")
                        hari = current_time.strftime("%A")
                        # Terjemahkan nama hari dan bulan ke Bahasa Indonesia
                        hari_indo = {
                            "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
                            "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
                        }
                        bulan_indo = {
                            "January": "Januari", "February": "Februari", "March": "Maret", "April": "April",
                            "May": "Mei", "June": "Juni", "July": "Juli", "August": "Agustus",
                            "September": "September", "October": "Oktober", "November": "November", "December": "Desember"
                        }
                        for eng, indo in hari_indo.items():
                            hari = hari.replace(eng, indo)
                        for eng, indo in bulan_indo.items():
                            tanggal = tanggal.replace(eng, indo)
                            waktu_lengkap = waktu_lengkap.replace(eng, indo)
                        # Tambahkan informasi waktu ke dalam prompt hanya jika user menanyakan waktu
                        context_prompt = f"""
                        CURRENT TIME INFORMATION:
                        - Currently it is: {hari}, {tanggal}, jam {jam} {timezone_label}
                        - Time: {jam}
                        - Date: {tanggal}
                        - Day: {hari}
                        - Timezone: {timezone_label}
                       IMPORTANT: Use the time information above when providing your answer. If a user asks for the current time or date, you MUST use the time data provided above, rather than suggesting a Google search.
                       USER QUESTIONS:
                        {user_prompt}
                        """
                    except Exception as e:
                        st.error(f"Error getting time information: {str(e)}")
                        context_prompt = user_prompt
                else:
                    # Jika tidak menanyakan tentang waktu, gunakan prompt user langsung
                    context_prompt = user_prompt
                # API key Gemini
                api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu
                # Gunakan endpoint yang benar untuk Gemini API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": context_prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 2048
                    }
                }
                try:
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        hasil = response.json()
                        try:
                            ai_jawaban = hasil["candidates"][0]["content"]["parts"][0]["text"]
                            st.success("Answer from Gemini:")
                            st.markdown(ai_jawaban)
                        except Exception as e:
                            st.error(f"There was an error reading the response from Gemini: {str(e)}")
                            st.code(hasil)
                    else:
                        st.error(f"Failed to contact Gemini API. Status code: {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"There is an error: {str(e)}")

elif choice == "Plant Disease Detection":
    st.header("🌱 Plant Disease Detection")
    try:
        model = load_model("models/plant_disease_cnn.h5")
    except Exception as e:
        st.error("Model not found! Make sure the model is in the 'models' folder.")
    
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Images", use_container_width=True)
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)
        prediction = model.predict(img_array)
        classes = ['Bacterial Spot', 'Healthy', 'Leaf Mold', 'Target Spot']
        max_index = np.argmax(prediction)
        
        if max_index < len(classes):
            result = classes[max_index]
            st.success(f"Prediction Results: {result}")
        else:
            st.error("Invalid prediction, check your model.")
        
        if result != "0":  # Hanya proses jika bukan kelas "0"
            if result == "Healthy":
                st.info("Healthy plant. No action required.")
            else:
                # Menggunakan AI Gemini untuk mendapatkan informasi tambahan
                prompt = f"Provide information about the causes and treatment methods for affected plants. {result}."
                api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 2048
                    }
                }
                try:
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        hasil = response.json()
                        try:
                            ai_jawaban = hasil["candidates"][0]["content"]["parts"][0]["text"]
                            st.warning(f"Causes and Treatment for {result}:")
                            st.info(ai_jawaban)
                        except Exception as e:
                            st.error(f"There was an error reading the response from Gemini: {str(e)}")
                            st.code(hasil)
                    else:
                        st.error(f"Failed to contact Gemini API. Status code:{response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"There is an error: {str(e)}")

elif choice == "Soil Type Detection":
    st.header("🪵 Soil Type Detection & Rekomendasi Pupuk")
    try:
        model = load_model("models/soil_classifier_cnn.h5")
    except Exception as e:
        st.error("Model not found! Make sure the model is in the 'models' folder.")
    
    uploaded_file = st.file_uploader("Upload Land Image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Images", use_container_width=True)
        
        # Preprocess gambar
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)
        
        # Prediksi menggunakan model
        prediction = model.predict(img_array)
        classes = ["0", "aluvial", "andosol", "chalk", "entisol", "humus", "inceptisol", "laterit", "sand"]
        fertile_soils = ["humus", "aluvial", "andosol", "inceptisol"]  # Tanah yang subur
        max_index = np.argmax(prediction)
        
        if max_index < len(classes):
            result = classes[max_index]
            st.success(f"Prediction Results: {result}")
        else:
            st.error("Invalid prediction, check your model.")
        
        if result != "0":  # Hanya proses jika bukan kelas "0"
            # Cek apakah jenis tanah subur atau tidak
            if result in fertile_soils:
                st.success(f"✅ {result.title()} is fertile soil and good for plantation!")
                # Prompt untuk tanah subur
                prompt = f"Provide information about the characteristics of {result} soil. Include its fertility attributes, ideal crops that grow well in it, and general maintenance tips to maintain its fertility. Format your answer with clear headings and bullet points."
            else:
                st.warning(f"⚠️ {result.title()} has lower fertility and needs treatment for optimal plant growth.")
                # Prompt untuk tanah tidak subur
                prompt = f"Provide information about the characteristics of {result} soil, why it has low fertility, and detailed steps on how to improve its fertility. Include specific fertilizer recommendations, soil amendments needed, and specific techniques to enhance its structure. Format your answer with clear headings and bullet points."
            
            api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048
                }
            }
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    hasil = response.json()
                    try:
                        ai_jawaban = hasil["candidates"][0]["content"]["parts"][0]["text"]
                        if result in fertile_soils:
                            st.info(f"Information for {result.title()} Soil:")
                            st.info(ai_jawaban)
                            
                            # Tambahkan rekomendasi khusus untuk tanah subur
                            st.success("Fertilizer Recommendations:")
                            st.markdown("""
                            For fertile soil like this, you only need to maintain its condition with:
                            - Light organic fertilizer application
                            - Regular organic matter addition
                            - Proper crop rotation
                            - Minimal soil disturbance
                            """)
                        else:
                            st.error(f"Soil Treatment Required for {result.title()}:")
                            st.info(ai_jawaban)
                            
                            # Tambahkan rekomendasi khusus untuk tanah tidak subur
                            st.warning("Critical Treatment Needed:")
                            st.markdown("""
                            This soil type requires significant intervention:
                            - Heavy fertilizer application
                            - Soil pH adjustment
                            - Organic matter incorporation
                            - Possible drainage improvement
                            - Regular soil testing
                            """)
                    except Exception as e:
                        st.error(f"There was an error reading the response from Gemini: {str(e)}")
                        st.code(hasil)
                else:
                    st.error(f"Failed to contact Gemini API. Status code: {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                st.error(f"There is an error: {str(e)}")

elif choice == "Harvest Prediction":
    st.header("🌾 Harvest Prediction (ton/ha)")
    try:
        # Load Model
        model = joblib.load("models/yield_prediction_pipeline.pkl")
        # Input
        region = st.selectbox("Region", ["Central", "East", "West", "South"])
        soil = st.selectbox("Soil Type", ["Clay", "Sandy", "Loam"])
        crop = st.selectbox("Crop", ["Rice", "Corn", "Wheat"])
        rainfall = st.number_input("Rainfall (mm)", min_value=0)
        temperature = st.number_input("Temperature (°C)", min_value=0)
        fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=0)
        irrigation = st.selectbox("Irrigation Used", ["Yes", "No"])
        weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy"])
        days = st.number_input("Days to Harvest", min_value=90)
        # Buat dataframe
        input_data = pd.DataFrame([{
            "Region": region,
            "Soil_Type": soil,
            "Crop": crop,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temperature,
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Weather_Condition": weather,
            "Days_to_Harvest": days
        }])
        if st.button("Prediction"):
            hasil = model.predict(input_data)[0]
            st.success(f"Estimated Harvest Results: {hasil:.2f} ton/ha")
    except Exception as e:
        st.error("Model not found! Make sure the model is in the 'models' folder.")