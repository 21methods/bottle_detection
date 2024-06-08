import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

# Set up paths for the model and image
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

# Minimum confidence threshold to filter weak detections
min_confidence = 0.2

# List of class labels MobileNet SSD was trained to detect
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Randomly generate bounding box colors for each class
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize the video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Set up the GPIO pin for the servo motor
servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50 Hz PWM frequency
servo.start(7.5)  # Initial position (90 degrees)

def move_servo_to_angle(angle):
    duty_cycle = 2.5 + (angle / 18.0)
    GPIO.output(servo_pin, True)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    servo.ChangeDutyCycle(0)

try:
    while True:
        ret, image = cap.read()
        if not ret:
            break

        height, width = image.shape[0], image.shape[1]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detected_objects = net.forward()

        for i in range(detected_objects.shape[2]):
            confidence = detected_objects[0, 0, i, 2]
            if confidence > min_confidence:
                class_index = int(detected_objects[0, 0, i, 1])
                if classes[class_index] == "bottle":
                    upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                    upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                    lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                    lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                    prediction_text = f"{classes[class_index]}: {confidence:.2f}"

                    cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
                    cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

                    # Move the servo motor to 180 degrees if a bottle is detected
                    move_servo_to_angle(180)
                    time.sleep(1)
                    # Move the servo motor back to 90 degrees
                    move_servo_to_angle(90)

        cv2.imshow('Detected Objects', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    # Cleanup the GPIO and video capture
    servo.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

# Set up paths for the model and image
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

# Minimum confidence threshold to filter weak detections
min_confidence = 0.2

# List of class labels MobileNet SSD was trained to detect
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Randomly generate bounding box colors for each class
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize the video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Set up the GPIO pin for the servo motor
servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50 Hz PWM frequency
servo.start(7.5)  # Initial position (90 degrees)

def move_servo_to_angle(angle):
    duty_cycle = 2.5 + (angle / 18.0)
    GPIO.output(servo_pin, True)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    servo.ChangeDutyCycle(0)

try:
    while True:
        ret, image = cap.read()
        if not ret:
            break

        height, width = image.shape[0], image.shape[1]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detected_objects = net.forward()

        for i in range(detected_objects.shape[2]):
            confidence = detected_objects[0, 0, i, 2]
            if confidence > min_confidence:
                class_index = int(detected_objects[0, 0, i, 1])
                if classes[class_index] == "bottle":
                    upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                    upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                    lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                    lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                    prediction_text = f"{classes[class_index]}: {confidence:.2f}"

                    cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
                    cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

                    # Move the servo motor to 180 degrees if a bottle is detected
                    move_servo_to_angle(180)
                    time.sleep(1)
                    # Move the servo motor back to 90 degrees
                    move_servo_to_angle(90)

        cv2.imshow('Detected Objects', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    # Cleanup the GPIO and video capture
    servo.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
