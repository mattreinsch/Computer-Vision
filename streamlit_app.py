import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import os

st.title("Computer Vision Showcase")
st.write("""
Welcome to the Computer Vision Showcase app! This application demonstrates various computer vision techniques that you can apply to your images. Here's how to use the app:

1. **Upload an Image**: Click on the "Choose an image..." button to upload a JPG, JPEG, or PNG file from your device.
2. **Select a Technique**: Choose a computer vision technique from the dropdown menu. The available options are:
   - **Edge Detection**: This technique highlights the edges within the image using the Canny edge detection algorithm.
   - **Object Detection**: This technique identifies and labels objects within the image using the YOLO (You Only Look Once) model.
   - **Blurring**: This technique applies a Gaussian blur to the image, smoothing out details.
3. **View Results**: Once you select a technique, the processed image will be displayed below the original image.

Feel free to experiment with different techniques and see how they transform your images!
""")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    img = Image.open(uploaded_file)
    img = np.array(img)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("Choose a Computer Vision technique to apply:")
    
    # Option to select different computer vision techniques
    option = st.selectbox(
        'Select a technique',
        ('Edge Detection', 'Object Detection', 'Blurring')
    )

    if option == 'Edge Detection':
        # Apply Canny edge detection
        edges = cv2.Canny(img, 100, 200)
        st.image(edges, caption='Edge Detection', use_column_width=True)

    elif option == 'Object Detection':
        try:
            # Download YOLO model files
            if not os.path.exists("yolov3.weights"):
                weights_url = "https://pjreddie.com/media/files/yolov3.weights"
                r = requests.get(weights_url, allow_redirects=True)
                open("yolov3.weights", 'wb').write(r.content)
            
            if not os.path.exists("yolov3.cfg"):
                cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
                r = requests.get(cfg_url, allow_redirects=True)
                open("yolov3.cfg", 'wb').write(r.content)
            
            if not os.path.exists("coco.names"):
                names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
                r = requests.get(names_url, allow_redirects=True)
                open("coco.names", 'wb').write(r.content)

            # Load YOLO
            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

            # Load class names
            with open("coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]

            # Prepare the image
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Show information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * img.shape[1])
                        center_y = int(detection[1] * img.shape[0])
                        w = int(detection[2] * img.shape[1])
                        h = int(detection[3] * img.shape[0])
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            st.image(img, caption='Object Detection', use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    elif option == 'Blurring':
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        st.image(blurred, caption='Blurred Image', use_column_width=True)