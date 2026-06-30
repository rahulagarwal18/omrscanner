<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
</p>

<h1 align="center">🚀 Live OMR Scanner System</h1>

<p align="center">
  <b>A real-time Optical Mark Recognition (OMR) scanner pipeline built with Python and OpenCV.</b><br>
  <i>Designed to process live video feeds, detect filled bubbles instantly, and export grades automatically.</i>
</p>

---

## 📖 Overview

Traditional OMR systems rely on expensive, proprietary hardware scanners. This project completely eliminates the need for hardware by utilizing **Computer Vision** to process live webcam feeds or scanned images in real-time.

By combining `OpenCV` for the image processing pipeline and `Flask` for the backend architecture, this system provides instantaneous test grading and automatically dumps the analytical results into a secure `SQLite` database.

## ✨ Key Features

- 📷 **Live Video Processing**: Scans and grades multiple-choice tests directly from a webcam stream.
- 📐 **Perspective Transformation**: Automatically detects paper edges and warps the perspective so the bubbles align perfectly, regardless of camera angle.
- 🧮 **Pixel Density Analysis**: Uses contour detection and binary thresholding to accurately determine which bubbles are shaded.
- 📊 **Automated Exporting**: Results are instantly graded against an answer key and pushed to a CSV or SQLite database.
- 🌐 **Web UI Integration**: Flask backend serves the processed video feed and results in real-time.

---

## 🛠️ System Architecture

1. **Grayscale Conversion**: Reduces image complexity.
2. **Gaussian Blurring**: Eliminates noise from cheap webcams.
3. **Canny Edge Detection**: Highlights the physical borders of the paper.
4. **Warp Perspective**: Flattens the image matrix.
5. **Masking & Thresholding**: Isolates the bubble rows and calculates non-zero pixels.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone https://github.com/rahulagarwal18/omrscanner.git

# Navigate to the directory
cd omrscanner

# Install the required dependencies
pip install opencv-python numpy flask
```

### Running the Scanner
```bash
python main.py
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/rahulagarwal18/omrscanner/issues).

---

<p align="center">
  Engineered with ❤️ by <a href="https://rahul-agarwal.in">Rahul Agarwal</a>
</p>
