#face_recognizer.py
import cv2
import json
import os

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 50:
            # Extract the name from the JSON file
            name = get_name_by_id(id)
            cv2.putText(img, f"{name} {confidence}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    return img

def get_name_by_id(id):
    if not os.path.isfile("user_data.json"):
        return "Unknown"

    with open("user_data.json", "r") as file:
        user_data = json.load(file)

    return user_data.get(str(id), "Unknown")

def recognize_face():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    # Set window size
    window_name = "Face Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allows window to be resized
    cv2.resizeWindow(window_name, 1000, 700)  # Set the window size to 1000x700 (width x height)

    while True:
        ret, img = video_capture.read()
        img = draw_boundary(img, faceCascade, 1.3, 6, (0, 255, 0), clf)
        cv2.imshow(window_name, img)

        if cv2.waitKey(1) == 13 or cv2.waitKey(1) == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()
