from deepface import DeepFace
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import datetime
import csv

# load webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# resize the window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# load the faces
data = 'C:/Users/Delsie/Desktop/projects/face_recognition_v2/data'

# load the model
model = YOLO('C:/Users/Delsie/Desktop/projects/face_recognition_v2/yolov8n-face_openvino_model')

now = datetime.datetime.now()
current_date = now.strftime('%Y-%m-%d')

# create a csv file to store the logs
with open(f'C:/Users/Delsie/Desktop/projects/face_recognition_v2/{current_date}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Date', 'Time'])


counter = 0


while True:
    ret, frame = cap.read()

    # detect for faces and crop them
    if ret:
        results: Results = model.predict(frame.copy(), imgsz=320, half=True, device='cpu', max_det=1)[0]
        detected_objects = []

        if hasattr(results, 'boxes') and hasattr(results, 'names'):
            for box in results.boxes.xyxy:
                object_id = int(box[-1])
                object_name = results.names.get(object_id)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                detected_objects.append((object_name, (x1, y1, x2, y2)))
        
        if counter % 10 == 0:
            # compare the face to the faces in the dataset
            for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):
                face = frame[y1:y2, x1:x2]
                name = 'Intruder'

                result = DeepFace.find(face, data, model_name='Facenet', distance_metric='euclidean_l2', enforce_detection=False, threshold=0.9)

                # get the name. draw bounding box and put the name on the frame
                if result[0].shape[0] != 0:
                    raw_name = result[0]['identity'][0].split('/')[-1]
                    name = raw_name.split('\\')[1]

                    # check if the name is already in the csv file
                with open(f'C:/Users/Delsie/Desktop/projects/face_recognition_v2/{current_date}.csv', 'r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    names = [row[0] for row in reader]
                    
                    if name not in names:
                        with open(f'C:/Users/Delsie/Desktop/projects/face_recognition_v2/{current_date}.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([name, current_date, now.strftime('%H:%M:%S')])

                color = (0, 0, 255) if name == 'Intruder' else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        counter += 1

        cv2.imshow('feed', frame)

    key = cv2.waitKey(1)
    if key == ord('q'): # press q to terminate the program
        break

cv2.destroyAllWindows()
