from detector import FaceDetector

FaceDetector = FaceDetector(result_path="results")
img_path = "images/1.jpg"
FaceDetector.inference(img_path=img_path)