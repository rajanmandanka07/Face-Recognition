import json
import os
from dataset_generator import generate_dataset, train_classifier
from face_recognizer import recognize_face

def main():
    while True:
        print("\nSelect an option:")
        print("1. Add new user and train classifier")
        print("2. Recognize face")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter the name: ")
            id = input("Enter the ID: ")

            # Check if the user already exists
            if user_exists(id):
                print(f"User with ID {id} already exists. Returning to the main menu.")
                continue

            generate_dataset(name, id)
            train_classifier("./facedata")
        elif choice == '2':
            recognize_face()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

def user_exists(id):
    if not os.path.isfile("user_data.json"):
        return False

    with open("user_data.json", "r") as file:
        user_data = json.load(file)

    return id in user_data

if __name__ == "__main__":
    main()
