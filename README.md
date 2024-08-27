# Face Recognition System

This project is a Face Recognition System that allows users to add new faces to a dataset, train a classifier on the dataset, and recognize faces using a pre-trained classifier. It consists of three main Python scripts:

1. `main.py` - The entry point for the system, providing a menu for adding new users, training the classifier, and recognizing faces.
2. `dataset_generator.py` - Handles dataset generation by capturing images from the webcam, preprocessing them, and saving them for training.
3. `face_recognizer.py` - Recognizes faces using the trained classifier and displays the recognized name and confidence level on the screen.

## Features

- **Add New User**: Capture images of a new user, add their details (name and ID) to the dataset, and update the classifier.
- **Train Classifier**: Automatically trains the classifier on the new dataset.
- **Recognize Face**: Uses the webcam to detect and recognize faces in real-time, displaying the recognized name on the screen.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PIL (Pillow)

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/facerecognition.git
   cd facerecognition
2. **Install the required Python packages**: 
    ```bash
    pip install opencv-python opencv-contrib-python numpy pillow
3. **Download the Haar Cascade XML file for face detection**:
    - You can download the haarcascade_frontalface_default.xml file from the OpenCv GitHub repository.
    - Place the file in the root directory of the project.
4. **Run the program**:
    ```bash
    python main.py

## Project Structure

- `main.py`: The main script that provides a command-line interface for adding new users, training the classifier, and recognizing faces.
- `dataset_generator.py`: Handles the dataset generation and training of the classifier.
- `face_recognizer.py`: Performs real-time face recognition using the webcam.
- `user_data.json`: A JSON file that stores user data (name and ID) for recognition purposes.
- `classifier.xml`: The trained classifier file used for face recognition.

## Usage

1. **Add New User**:
   - Run the script using `python main.py`.
   - Choose option `1` to add a new user.
   - Enter the name and ID of the user.
   - The system will guide you through capturing 30 images of the user's face. Follow the on-screen instructions.
   - The images will be saved, and the classifier will be trained automatically.

2. **Recognize Face**:
   - Run the script using `python main.py`.
   - Choose option `2` to recognize a face.
   - The system will use the webcam to detect and recognize faces in real-time, displaying the name and confidence level on the screen.

3. **Exit**:
   - Choose option `3` to exit the program.

## JSON Data Storage

- The user data (name and ID) is stored in a `user_data.json` file as key-value pairs. The key is the ID, and the value is the name of the user.

## Important Notes

- Ensure that your webcam is connected and working properly before running the program.
- The system requires good lighting conditions to perform accurate face recognition.
- The confidence level of recognition depends on the quality and number of images in the dataset.

## License

This project is licensed under the MIT License.

## Author

- [Rajankumar Mandanka](https://github.com/rajanmandanka07/Face-Recognition.git)

## Acknowledgments

- This project uses the OpenCV library for face detection and recognition.
- The Haar Cascade classifier used in this project is provided by OpenCV.
