import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

servo_pin = 18
ir_sensor_pin = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(ir_sensor_pin, GPIO.IN)

servo = GPIO.PWM(servo_pin, 50)
servo.start(7.5)  # Initialize the servo to the original position (90 degrees)

def move_servo_to_angle(angle):
    duty_cycle = 2.5 + (angle / 18.0)
    GPIO.output(servo_pin, True)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    servo.ChangeDutyCycle(0)

# Calibrate the IR sensor
def calibrate_ir_sensor():
    print("Calibrating IR sensor...")
    time.sleep(2)
    print("Place a plastic bottle in front of the sensor and press Enter")
    input()
    plastic_reading = GPIO.input(ir_sensor_pin)
    print(f"Plastic reading: {plastic_reading}")

    print("Place a non-plastic object in front of the sensor and press Enter")
    input()
    non_plastic_reading = GPIO.input(ir_sensor_pin)
    print(f"Non-plastic reading: {non_plastic_reading}")

    threshold = (plastic_reading + non_plastic_reading) / 2
    print(f"Threshold: {threshold}")
    return threshold

# Use the calibrated threshold to determine if the object is plastic or not
def is_plastic(threshold):
    return GPIO.input(ir_sensor_pin) > threshold

try:
    threshold = calibrate_ir_sensor()
    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read image from camera")
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

                    # Check if the detected bottle is plastic using the IR sensor
                    if is_plastic(threshold):
                        print("Plastic bottle detected")
                        # Move the servo to 180 degrees for a plastic bottle
                        move_servo_to_angle(180)
                    else:
                        print("Non-plastic bottle detected")
                        # Keep the servo at its original position (90 degrees)
                        move_servo_to_angle(90)

        # Resize the image before displaying it
        resized_image = cv2.resize(image, (640, 480))  # Change the resolution as needed

        cv2.imshow('Detected Objects', resized_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    servo.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
