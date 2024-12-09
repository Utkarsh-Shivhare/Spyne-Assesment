# Spyne Assignment

## Project 1: Image Background Changer

This project implements automated background replacement for images. The `project_1.py` file contains functions that systematically process and modify image backgrounds.

### Testing Process
The testing folder documents the step-by-step approach used to achieve accurate background replacement results.

### Results
Output images are available in the assignment folder:
- 1.jpg through 6.jpg demonstrate various background replacement scenarios

## Project 2: Car Angle Detection API

Flask API that predicts car angles from images using EfficientNet model.

### Setup

1. Clone repository
```bash
git clone https://github.com/Utkarsh-Shivhare/Spyne-Assesment.git
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download model
- Place `best_model.pth` in project root directory

### Project Structure
```
project_2/
├── templates/
│   └── index.html
├── app.py
├── best_model.pth
├── requirements.txt
├── document.txt
```

### Usage

1. Start Flask server
```bash
python app.py
```

2. Access API
- Web interface: http://localhost:5000
- POST endpoint: http://localhost:5000/predict
  - Send image file with key 'image'
  - Returns JSON with angle and confidence

### API Response Format
```json
{
    "angle": "90",
    "confidence": 0.95
}
```

### Supported Angles
0°, 40°, 90°, 130°, 180°, 230°, 270°, 320°