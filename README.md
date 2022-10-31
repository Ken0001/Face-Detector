# Face-Detector
* Idea from https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
* Support ONNX Model

## Installation
1. Download Repository

    ```$ git clone https://github.com/Ken0001/Face-Detector.git```

2. Install Python Package

    ```$ pip install -r requirements.txt```

---
## Usage
```python
from detector import FaceDetector

FaceDetector = FaceDetector()
FaceDetector.inference(img_path="your/image/path")
```

* You can change result path or other argument when initial constructor, please see line 14 in ```detector.py```
```python
FaceDetector = FaceDetector(result_path="results")
```
---
## Result
* Terminal Output
```
$ python3 detector.py --img_path images/1.jpg
[Warning]...
.
.
.
Image: 1.jpg
 - Inference time: 0.0061283111572265625 s
 - Face 1: [250  55 274  83]
 - Face 2: [300 157 326 187]
 - Face 3: [484  56 505  81]
Total 3 faces.
```

* Example Result

![img2](https://github.com/Ken0001/Face-Detector/blob/main/1.jpg)
