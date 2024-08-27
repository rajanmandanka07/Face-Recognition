import cv2
import os
from PIL import Image
import numpy as np
import json

def generate_dataset(name, id):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h + 20, x:x + w + 15]
        return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0

    # Instruction to display
    instruction = "Look into the frame"
    photos_to_take = 30

    # Consistently resize the window
    window_size = (800, 600)

    # Define the rectangle position and size
    rect_start_point = (200, 100)  # Top-left corner of the rectangle
    rect_end_point = (600, 500)    # Bottom-right corner of the rectangle
    rect_color = (0, 255, 0)       # Green color
    rect_thickness = 2             # Thickness of the rectangle border

    # Show the instruction with a 5-second timer
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, window_size)  # Ensure consistent window size
            # Draw the rectangle on the frame
            cv2.rectangle(frame, rect_start_point, rect_end_point, rect_color, rect_thickness)
            cv2.putText(frame, f"{instruction} in {i} seconds...", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped face", frame)
            cv2.waitKey(1000)  # Wait for 1 second
        else:
            break

    # Capture photos
    for _ in range(photos_to_take):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, window_size)  # Ensure consistent window size
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"facedata/{name}.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
            cv2.imshow("Cropped face", frame)  # Display the frame
        if cv2.waitKey(1) == 27:  # Use Escape key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")

    # Add user to JSON file
    add_user_to_json(name, id)

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    print("Training successful")

def add_user_to_json(name, id):
    user_data = {}
    if os.path.isfile("user_data.json"):
        with open("user_data.json", "r") as file:
            user_data = json.load(file)

    user_data[id] = name

    with open("user_data.json", "w") as file:
        json.dump(user_data, file)
