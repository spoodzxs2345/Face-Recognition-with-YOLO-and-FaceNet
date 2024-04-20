from deepface import DeepFace
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import datetime
import csv
import os

# load webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# resize the window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# User Interface
img_bg = cv2.imread('C:/Users/Delsie/Desktop/projects/face_recognition_v2/UE Attendance/System Background.png')
folder_mode_path = 'C:/Users/Delsie/Desktop/projects/face_recognition_v2/UE Attendance/modes'
mode_list = os.listdir(folder_mode_path)
img_mode_list = []
for path in mode_list:
    img_mode_list.append(cv2.imread(os.path.join(folder_mode_path, path)))

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

mode_type = 0

counter = 0


while True:
    ret, frame = cap.read()

    img_bg[162:162+480, 55:55+640] = frame
    img_bg[44:44+633, 808:808+414] = img_mode_list[mode_type]

    # detect for faces and crop them
    if ret:
        results: Results = model.predict(img_bg.copy(), imgsz=320, half=True, device='cpu', max_det=1)[0]
        detected_objects = []

        if hasattr(results, 'boxes') and hasattr(results, 'names'):
            for box in results.boxes.xyxy:
                object_id = int(box[-1])
                object_name = results.names.get(object_id)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                detected_objects.append((object_name, (x1, y1, x2, y2)))
        
        if len(detected_objects) != 0: # check for faces
            # compare the face to the faces in the dataset
            for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):
                face = img_bg[y1:y2, x1:x2]
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

                            if name != 'Intruder':
                                # show image and name on the screen
                                mode_type = 1
                                img_bg[44:44+633, 808:808+414] = img_mode_list[mode_type]
                                (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                                offset = (414 - w) // 2
                                cv2.putText(img_bg, name, (808 + offset, 445), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1)
                                img_attendance = cv2.imread(result[0]['identity'][0])
                                img_resize = cv2.resize(img_attendance, (216, 216))
                                img_bg[175:175 + 216, 909:909 + 216] = img_resize


                            secondsElapsed = (datetime.datetime.now() - now).total_seconds()
                            
                            if secondsElapsed > 30:
                                mode_type = 2


                    else:
                        mode_type = 3

                    color = (0, 0, 255) if name == 'Intruder' else (0, 255, 0)

                    cv2.rectangle(img_bg, (x1, y1), (x2, y2), color, 2)
                    #cv2.putText(img_bg, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            mode_type = 0

        cv2.imshow('attendance', img_bg)
        #cv2.imshow('feed', frame)

    key = cv2.waitKey(1)
    if key == ord('q'): # press q to terminate the program
        break

cv2.destroyAllWindows()
