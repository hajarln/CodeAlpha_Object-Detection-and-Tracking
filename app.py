import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Object Tracking Dashboard", 
    layout="wide",
    page_icon="📊"
)

# Injection CSS : Ajustements précis des couleurs
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdf6f0 0%, #f5efff 100%);
        color: #3a3a4f;
    }
    
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    [data-testid="stImage"] img {
        max-height: 450px !important;
        width: auto !important;
        object-fit: contain !important;
        border-radius: 16px;
        box-shadow: 0px 10px 30px rgba(180, 160, 180, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    h1 {
        color: #000000 !important;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    h3 {
        color: #5c5470 !important;
        font-weight: 700 !important;
    }
    
    /* Barre latérale (Sidebar) - Tons Lilas/Gris doux & Bordure Verte Amande */
    [data-testid="stSidebar"] {
        background-color: #f7f3f9 !important;
        border-right: 3px solid #f6a192;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #5c5470 !important;
        font-weight: 600;
    }
    
    .instructions-box {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(5px);
        border-left: 5px solid #f6a192; /* Orange clair */
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(220, 200, 200, 0.1);
    }
    .instructions-title {
        color: #e07a5f !important;
        font-weight: bold;
        font-size: 1.05rem;
        margin-bottom: 4px;
    }
    .instructions-text {
        color: #6c5b7b !important;
        font-size: 0.95rem;
        line-height: 1.4;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important; /* Chiffres (0) passés en noir */
        font-size: 2.4rem !important;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        color: #7d84b2 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 16px;
        border-radius: 16px;
        border-left: 5px solid #f6a192;
        border-top: 1px solid rgba(255, 255, 255, 0.5);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0px 6px 20px rgba(180, 160, 180, 0.1);
    }
    
    .stMultiSelect div[data-baseweb="select"] {
        background-color: white !important;
        border-radius: 10px;
    }
    span[data-baseweb="tag"] {
        background-color: #f6a192 !important; /* Même orange clair que le guide */
        color: white !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] role[button], span[data-baseweb="tag"] svg {
        fill: white !important;
        color: white !important;
    }
    
    
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #f6a192 !important; /* Bulle du slider */
        border-color: #f6a192 !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #f6a192 !important;
        font-weight: 700 !important;
    }
   [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #5c5470 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.title(" Web Application for Object Detection & Tracking")

st.sidebar.header("⚙️ Project Settings")

uploaded_file = st.sidebar.file_uploader(
    "📂 Step 1: Choose a video (MP4, AVI, MOV)", 
    type=["mp4", "avi", "mov"]
)

class_mapping = {
    "Person": 0, "Bicycle": 1, "Car": 2, "Motorcycle": 3, "Airplane": 4, "Bus": 5, "Train": 6, "Truck": 7, "Boat": 8,
    "Traffic light": 9, "Fire hydrant": 10, "Stop sign": 11, "Parking meter": 12, "Bench": 13, "Bird": 14,
    "Cat": 15, "Dog": 16, "Horse": 17, "Sheep": 18, "Cow": 19, "Elephant": 20, "Bear": 21, "Zebra": 22, "Giraffe": 23,
    "Backpack": 24, "Umbrella": 25, "Handbag": 26, "Tie": 27, "Suitcase": 28, "Frisbee": 29, "Skis": 30,
    "Snowboard": 31, "Sports ball": 32, "Kite": 33, "Baseball bat": 34, "Baseball glove": 35,
    "Skateboard": 36, "Surfboard": 37, "Tennis racket": 38, "Bouteille": 39, "Wine glass": 40,
    "Cup": 41, "Fork": 42, "Knife": 43, "Spoon": 44, "Bowl": 45, "Banana": 46, "Apple": 47, "Sandwich": 48,
    "Orange": 49, "Broccoli": 50, "Carrot": 51, "Hot dog": 52, "Pizza": 53, "Donut": 54, "Cake": 55, "Chair": 56,
    "Couch": 57, "Potted plant": 58, "Bed": 59, "Dining table": 60, "Toilet": 61, "TV": 62,
    "Laptop": 63, "Mouse": 64, "Remote": 65, "Keyboard": 66, "Cell phone": 67,
    "Microwave": 68, "Oven": 69, "Toaster": 70, "Sink": 71, "Refrigerator": 72, "Book": 73,
    "Clock": 74, "Vase": 75, "Scissors": 76, "Teddy bear": 77, "Hair drier": 78, "Toothbrush": 79
}

selected_classes = st.sidebar.multiselect(
    "🎯 Step 2: Classes to track:",
    options=list(class_mapping.keys()),
    default=["Person", "Book", "Bicycle"]
)
classes_to_track = [class_mapping[cls] for cls in selected_classes]

conf_threshold = st.sidebar.slider(
    "🔍 Confidence Threshold", 
    min_value=0.1, max_value=1.0, value=0.25, step=0.05
)

col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("""
        <div class="instructions-box">
            <div class="instructions-title">💡 Quick User Guide:</div>
            <div class="instructions-text">
                1. Choose your classes on the left >>. &nbsp;|&nbsp; 
                2. Drop your video file. &nbsp;|&nbsp; 
                3. <b>The analysis starts automatically!</b>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("### 🎥 Live Video Stream")
    video_placeholder = st.empty()

with col2:
    st.write("### 📈 Statistics")
    st.write("")
    stat_unique_objects = st.empty()
    st.write("")
    st.write("")
    stat_current_objects = st.empty()

stat_unique_objects.metric(label="Total Unique Objects Tracked", value="0")
stat_current_objects.metric(label="Objects Present on Screen", value="0")

if uploaded_file is not None:
    if len(selected_classes) == 0:
        st.sidebar.error("❌ Please select at least one class!")
    else:
        with st.spinner("Loading YOLOv8 model..."):
            model = YOLO("yolov8n.pt")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name
        
        cap = cv2.VideoCapture(tfile_path)
        tracked_ids = set()

        status_message = st.info("🏃‍♂️ Automatic analysis in progress...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame, 
                persist=True, 
                conf=conf_threshold, 
                classes=classes_to_track, 
                verbose=False
            )

            current_count = 0
            if results[0].boxes is not None and results[0].boxes.id is not None:
                current_ids = results[0].boxes.id.int().tolist()
                current_count = len(current_ids)
                for obj_id in current_ids:
                    tracked_ids.add(obj_id)

            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(annotated_frame_rgb, channels="RGB")
            
            stat_unique_objects.metric(
                label="Total Unique Objects Tracked", 
                value=str(len(tracked_ids))
            )
            stat_current_objects.metric(
                label="Objects Present on Screen", 
                value=str(current_count)
            )

        cap.release()
        status_message.empty()
        st.success("🎯 Analysis completed!")
        
        try:
            if os.path.exists(tfile_path):
                os.unlink(tfile_path)
        except Exception as e:
            pass
else:
    st.info("💡 Awaiting a video file upload to start real-time analysis.")