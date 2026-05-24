import cv2
import os
import sys
import warnings
import logging
from ultralytics import YOLO

warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

def main():

    video_source = "video.mp4" 
    
    if isinstance(video_source, str) and not os.path.exists(video_source):
        print(f"Error: The file '{video_source}' was not found.")
        print("Please ensure your video file is placed in this directory:")
        print(os.getcwd())
        print("Alternatively, change 'video_source = 0' to use your webcam.")
        sys.exit()

    print(f"Loading video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit()

    print("Loading pre-trained YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    print("\n" + "="*50)
    print("🚀 Object Detection & Tracking Started!")
    print("👉 Press 'q' on the video window to quit.")
    print("="*50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

   
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        current_frame_objects = 0
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            tracking_ids = results[0].boxes.id.int().tolist()
            current_frame_objects = len(tracking_ids)

        annotated_frame = results[0].plot()

        cv2.putText(
            annotated_frame, 
            f"Objects on Screen: {current_frame_objects}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0),  
            2, 
            cv2.LINE_AA
        )

        resized_frame = cv2.resize(annotated_frame, (960, 540))

        cv2.imshow("YOLOv8 Real-Time Object Tracking", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System stopped successfully.")

if __name__ == "__main__":
    main()