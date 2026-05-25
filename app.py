import streamlit as st
import cv2
import os
import tempfile
import torch
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Object Tracking Dashboard",
    layout="wide",
    page_icon="📊"
)

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

    [data-testid="stSidebar"] {
        background-color: #f7f3f9 !important;
        border-right: 3px solid #f6a192;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #5c5470 !important;
        font-weight: 600;
    }

    .instructions-box {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(5px);
        border-left: 5px solid #f6a192;
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
        color: #000000 !important;
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
        box-shadow: 0px 6px 20px rgba(180, 160, 180, 0.1);
    }

    .stMultiSelect div[data-baseweb="select"] {
        background-color: white !important;
        border-radius: 10px;
    }

    span[data-baseweb="tag"] {
        background-color: #f6a192 !important;
        color: white !important;
        border-radius: 6px !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #f6a192 !important;
        border-color: #f6a192 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Web Application for Object Detection & Tracking")

st.sidebar.header("⚙️ Project Settings")

uploaded_file = st.sidebar.file_uploader(
    "📂 Step 1: Choose a video (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"]
)

class_mapping = {
"Person": 0, "Bicycle": 1, "Car": 2, "Motorcycle": 3, "Airplane": 4, "Bus": 5, "Train": 6, "Truck": 7, "Boat": 8,
    "Traffic light": 9,"Stop sign": 11, "Parking meter": 12, "Bench": 13, "Bird": 14,
    "Cat": 15, "Dog": 16, "Horse": 17, "Sheep": 18,  "Elephant": 20, "Bear": 21, "Giraffe": 23,
    "Backpack": 24, "Umbrella": 25, "Handbag": 26, "Tie": 27, "Suitcase": 28, "Frisbee": 29, "Skis": 30,
    "Snowboard": 31, "Sports ball": 32, "Kite": 33, "Baseball bat": 34, "Baseball glove": 35,
    "Skateboard": 36, "Surfboard": 37, "Tennis racket": 38, "Bouteille": 39, "Wine glass": 40,
    "Cup": 41, "Fork": 42, "Knife": 43, "Spoon": 44, "Bowl": 45, "Banana": 46, "Apple": 47, "Sandwich": 48,
    "Orange": 49, "Broccoli": 50, "Carrot": 51, "Hot dog": 52, "Pizza": 53, "Donut": 54, "Cake": 55, "Chair": 56,
    "Couch": 57, "Potted plant": 58, "Bed": 59, "Dining table": 60, "Toilet": 61, "TV": 62,
    "Laptop": 63, "Mouse": 64, "Remote": 65, "Keyboard": 66, "Cell phone": 67,
     "Sink": 71, "Refrigerator": 72, "Book": 73,
    "Clock": 74, "Vase": 75 
}

selected_classes = st.sidebar.multiselect(
    "🎯 Step 2: Classes to track:",
    options=list(class_mapping.keys()),
    default=["Person", "Book", "Car"]
)

classes_to_track = [class_mapping[cls] for cls in selected_classes]

conf_threshold = st.sidebar.slider(
    "🔍 Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

col1, col2 = st.columns([2.5, 1])

with col1:

    st.markdown("""
        <div class="instructions-box">
            <div class="instructions-title">
                💡 Quick User Guide:
            </div>

            
                1. Choose your classes on the left >> | 2. Upload your video | 3. Analysis starts automatically
          
        </div>
    """, unsafe_allow_html=True)

    st.write("### 🎥 Live Video Stream")

    video_placeholder = st.empty()

with col2:

    st.write("### 📈 Statistics")

    stat_unique_objects = st.empty()

    st.write("")

    stat_current_objects = st.empty()

stat_unique_objects.metric(
    label="Total Unique Objects Tracked",
    value="0"
)

stat_current_objects.metric(
    label="Objects Present on Screen",
    value="0"
)

if uploaded_file is not None:

    if len(selected_classes) == 0:

        st.sidebar.error("❌ Please select at least one class!")

    else:

        with st.spinner("Loading YOLOv8 model..."):

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = YOLO("yolov8n.pt")

            model.to(device)

            st.sidebar.success(f"🚀 Running on: {device.upper()}")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp4'
        ) as tfile:

            tfile.write(uploaded_file.read())

            tfile_path = tfile.name

        cap = cv2.VideoCapture(tfile_path)

        tracked_ids = set()

        status_message = st.info(
            "🏃‍♂️ Automatic analysis in progress..."
        )

        frame_count = 0

        FRAME_SKIP = 2

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (640, 360))

            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                classes=classes_to_track,
                verbose=False,
                imgsz=640
            )

            current_count = 0

            if (
                results[0].boxes is not None
                and results[0].boxes.id is not None
            ):

                current_ids = (
                    results[0]
                    .boxes
                    .id
                    .int()
                    .tolist()
                )

                current_count = len(current_ids)

                for obj_id in current_ids:
                    tracked_ids.add(obj_id)

            annotated_frame = results[0].plot()

            annotated_frame_rgb = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            if frame_count % 4 == 0:

                video_placeholder.image(
                    annotated_frame_rgb,
                    channels="RGB",
                     width="stretch"
                )

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

        except Exception:
            pass

else:

    st.info(
        "💡 Awaiting a video file upload to start real-time analysis."
    )