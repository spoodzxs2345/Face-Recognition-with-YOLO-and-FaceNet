from deepface import DeepFace
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

data = 'C:/Users/Delsie/Desktop/projects/face_recognition_v2/data'
img = cv2.imread('C:/Users/Delsie/Desktop/projects/face_recognition_v2/test_img_3.jpg')
test_img = cv2.resize(img, (640, 480))
model = YOLO('C:/Users/Delsie/Desktop/projects/face_recognition_v2/yolov8n-face_openvino_model')

results: Results = model.predict(test_img.copy(), imgsz=320, half=True, device='cpu', max_det=1)[0]
detected_objects = []

if hasattr(results, 'boxes') and hasattr(results, 'names'):
    for box in results.boxes.xyxy:
        object_id = int(box[-1])
        object_name = results.names.get(object_id)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        detected_objects.append((object_name, (x1, y1, x2, y2)))
        
        # compare the face to the faces in the dataset
    for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):
        face = test_img[y1:y2, x1:x2]
        name = 'Intruder'

        result = DeepFace.find(face, data, model_name='Facenet', distance_metric='euclidean_l2', enforce_detection=False, threshold=0.9)

        # get the name. draw bounding box and put the name on the test_img
        if result[0].shape[0] != 0:
            raw_name = result[0]['identity'][0].split('/')[-1]
            name = raw_name.split('\\')[1]

        color = (0, 0, 255) if name == 'Intruder' else (0, 255, 0)

        cv2.rectangle(test_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(test_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imshow('test', test_img)
cv2.resizeWindow('test', 640, 480)
cv2.waitKey(0)
cv2.destroyAllWindows()