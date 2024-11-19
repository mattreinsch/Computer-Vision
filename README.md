
```markdown
# Computer Vision Showcase

Welcome to the Computer Vision Showcase app! This application demonstrates various computer vision techniques that you can apply to your images.

## Features

- **Edge Detection**: Highlights the edges within the image using the Canny edge detection algorithm.
- **Object Detection**: Identifies and labels objects within the image using the YOLO (You Only Look Once) model.
- **Blurring**: Applies a Gaussian blur to the image, smoothing out details.

## How to Use

1. **Upload an Image**: Click on the "Choose an image..." button to upload a JPG, JPEG, or PNG file from your device.
2. **Select a Technique**: Choose a computer vision technique from the dropdown menu. The available options are:
   - **Edge Detection**
   - **Object Detection**
   - **Blurring**
3. **View Results**: Once you select a technique, the processed image will be displayed below the original image.

## Installation

To run this app locally, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/computer-vision-showcase.git
   cd computer-vision-showcase
   ```

2. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the YOLO files**:
   - yolov3.weights
   - yolov3.cfg
   - coco.names

   Place these files in the same directory as your script.

4. **Run the app**:
   ```sh
   streamlit run streamlit_app.py
   ```

## Dependencies

- `streamlit`
- `opencv-python`
- `numpy`
- `Pillow`

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- YOLO (You Only Look Once)
- Streamlit
- OpenCV

```
