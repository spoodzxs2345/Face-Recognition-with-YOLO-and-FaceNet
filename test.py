from deepface import DeepFace
import cv2

data = 'C:/Users/Delsie/Desktop/projects/face_recognition_v2/data'
test_img = cv2.imread('C:/Users/Delsie/Desktop/projects/face_recognition_v2/test_img_1.jpg')

result = DeepFace.find(test_img, data, model_name='Facenet', distance_metric='cosine', enforce_detection=False)
print(result[0])
#print(result[0]['identity'][0])
#print(result[0]['identity'][0].split('/')[-1])
print(result[0].shape)

if result[0].shape[0] != 0:
    raw_name = result[0]['identity'][0].split('/')[-1]
    name = raw_name.split('\\')[1]

    print(name)
    print(type(name))