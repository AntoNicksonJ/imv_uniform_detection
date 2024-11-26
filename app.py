import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from collections import Counter

# Set custom background


def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Video processing functions


def process_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Perform inference
        annotated_frame = results[0].plot()  # Annotate the frame
        out.write(annotated_frame)  # Write the frame to the output video

    cap.release()
    out.release()


def process_realtime_video(model):
    cap = cv2.VideoCapture(0)  # Open the webcam
    stframe = st.empty()  # Placeholder for displaying the video stream
    class_counter = st.empty()  # Placeholder for displaying the class counts

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Perform inference
        detections = results[0]  # Get detections from the first result

        # Count detected classes dynamically
        class_counts = Counter([detections.names[int(cls)]
                               for cls in detections.boxes.cls])

        # Display class counts in Streamlit
        with class_counter.container():
            st.write("### Real-Time Class Count")
            for cls, count in class_counts.items():
                st.write(f"- *{cls}*: {count}")

        # Convert the frame to RGB for Streamlit
        annotated_frame = detections.plot()  # Annotate the frame
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()

# Streamlit App


def main():
    # Set the background image
    set_bg_image("https://png.pngtree.com/thumb_back/fw800/background/20231219/pngtree-nanosensors-unveiled-abstract-visualization-of-molecular-detection-background-image_15518095.jpg")

    # App title and sidebar
    st.title("🎥 Uniform Complaince Detector")
    st.sidebar.title("⚙ Options")

    # Load YOLO model
    model_path = st.sidebar.text_input(
        "📁 Model Path", r"D:\From_8_24\IMV\imv_uniform_detection\final_model.pt")
    model = YOLO(model_path)  # Load the YOLOv8 model

    # Mode selection
    mode = st.sidebar.selectbox(
        "🚀 Choose Mode", ["Upload Video", "Real-Time Detection"])

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("📤 Upload a video file", type=[
                                         "mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.close()

            output_path = f"{os.path.splitext(temp_file.name)[0]}_output.mp4"

            st.write("🔄 Processing video...")
            process_video(model, temp_file.name, output_path)

            st.video(output_path)
            st.success("✅ Processing complete! Check the output video above.")

    elif mode == "Real-Time Detection":
        st.write("📹 Starting real-time detection...")
        process_realtime_video(model)


if __name__ == "__main__":
    main()
