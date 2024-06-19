import streamlit as st
import cv2
from PIL import Image
import numpy as np
from yolo import yolo_object,yolo_object_video
import argparse
import time
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData.yml")
def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    name='Unknown'
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        id, uncertainty = rec.predict(gray[y:y + h, x:x + w])
        print(id, uncertainty)

        if (uncertainty< 40):
            if (id == 1 or id == 3 or id == 5):
                name = "Nachiketa"
                cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
        else:
            cv2.putText(img, ' ', (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
    return img




def main():
    st.title("Face Recognition WebApp")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    file_type = st.radio("Select file type:", ("Image", "Video"), key="file_type_radio")

    if file_type == "Image":
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image, use_column_width=True)
            if st.button("Recognise"):
                progress_bar = st.progress(0)
                progress_bar.progress(20)  # Update progress bar to indicate image processing
                result_img_1 = detect_faces(our_image)
                progress_bar.progress(50)  # Update progress bar to indicate processing progress
                image = np.array(our_image)
                result_img_2 = yolo_object(image)  # Assuming this function returns processed image
                progress_bar.progress(100)  # Update progress bar to indicate processing completion
                combined_image = cv2.addWeighted(result_img_1, 0.5, result_img_2, 0.5, 0)
                st.image(combined_image, use_column_width=True)

    elif file_type == "Video":
        video_file = st.file_uploader("Upload Video", type=['mp4'])
        if video_file is not None:
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            
            video = cv2.VideoCapture(video_path)
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            progress_bar = st.progress(0)
            while True:
                ret, frame = video.read()
                if not ret:
                    break  
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result_img_1 = detect_faces(pil_image)
                np_image = np.array(pil_image)
                #result_img_2 = yolo_object_video(np_image)  # Assuming this function returns processed image
                #combined_image = cv2.addWeighted(result_img_1, 0.5, result_img_2, 0.5, 0)
                out.write(result_img_1)
                frame_count += 1
                progress_percentage = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress_percentage)  # Update progress bar
            st.success("Video processing completed!")
            video.release()
            out.release()


        



if __name__ == '__main__':
    main()
