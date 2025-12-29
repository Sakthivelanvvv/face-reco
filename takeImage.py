import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time

def take_image(enrollment, name, haar_cascade_path, train_image_path, message, err_screen, text_to_speech):
    if not enrollment or not name:
        error_message = "Please enter both your Enrollment Number and Name."
        text_to_speech(error_message)
        return

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise ValueError("Unable to open camera")

        detector = cv2.CascadeClassifier(haar_cascade_path)
        sample_num = 0
        directory = f"{enrollment}_{name}"
        path = os.path.join(train_image_path, directory)

        if os.path.exists(path):
            raise FileExistsError("Student data already exists")

        os.mkdir(path)

        while True:
            ret, img = cam.read()
            if not ret:
                raise ValueError("Failed to capture image from camera")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sample_num += 1
                image_path = os.path.join(path, f"{name}_{enrollment}_{sample_num}.jpg")
                cv2.imwrite(image_path, gray[y:y+h, x:x+w])
                cv2.imshow("Frame", img)

            if cv2.waitKey(1) & 0xFF == ord("q") or sample_num > 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        row = [enrollment, name]
        with open("StudentDetails/studentdetails.csv", "a+", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(row)

        success_message = f"Images saved for ER No: {enrollment} Name: {name}"
        message.configure(text=success_message)
        text_to_speech(success_message)

    except FileExistsError as e:
        error_message = "Student data already exists"
        text_to_speech(error_message)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        text_to_speech(error_message)

# Example usage
take_image("12345", "John Doe", "path/to/haarcascade.xml", "path/to/train_images", message_label, error_label, text_to_speech_function)