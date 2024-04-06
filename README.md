# Face Recognition using YOLOv8 and FaceNet

> [!IMPORTANT]
> Make sure to create a virtual environment first before running the program to avoid conflicts with dependencies

## How to run the project
1. Clone this repository
   ```
   
   git clone https://github.com/spoodzxs2345/Face-Recognition-with-YOLO-and-FaceNet.git
   
   ```
2. Install the dependencies
   > Note: I used Python 3.10 for this project
   ```
   
   pip install -r requirements.txt
   
   ```
3. Update the file paths in [face_recognition.py](face_recognition.py) (*the ones in the `data` and `model` variable*)
4. Run the program
   > Note: It will download the weights for Facenet in the first run so I recommend you run [test.py](test.py) first.
   ```
   
   python face_recognition.py
   
   ```

## How the project was done
1. One of the required features for this project is for it to be able to detect faces even with face mask. Since the other faces don't wear face mask, they were needed to be augmented. To do that, I used a Python script to put face mask on the faces.
2.  After the faces were augmented, I did setup the model for face detection. I used a [pre-trained YOLOv8 face detection model](https://github.com/akanametov/yolov8-face) for face detection and to ensure that the detection will be accurate. I then exported the model to OpenVino since it had the highest FPS during inference for CPU.
3.  I extracted the faces from the dataset, cropped them, and put them into a data folder.
4.  I then built the program for real-time face recognition. I used the Python library called [DeepFace](https://github.com/serengil/deepface).

## References
- [YOLOv8 Face Detection](https://github.com/akanametov/yolov8-face)
- [DeepFace](https://github.com/serengil/deepface)
- [Face Recognition using YOLOv8](https://github.com/sOR-o/Face-Recognition)
