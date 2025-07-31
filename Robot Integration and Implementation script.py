import RPi.GPIO as GPIO

import time

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np

import cv2



# GPIO and Servo Pins

BASE_PIN = 17

SHOULDER_PIN = 27

ELBOW_PIN = 22

WRIST_PIN = 5

GRIPPER_PIN = 6



IN1 = 23

IN2 = 24

IN3 = 25

IN4 = 16

ENA = 12

ENB = 13



# Waste classification model setup

model = load_model('Waste_Classification_Model.h5')

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']



# Bin durations for movement

BIN_DURATIONS = {

    'cardboard': 1,

    'glass': 2,

    'metal': 3,

    'paper': 4,

    'plastic': 5,

    'trash': 6

}



# Servo setup function

def setup_servo(pin):

    GPIO.setup(pin, GPIO.OUT)

    pwm = GPIO.PWM(pin, 50)  # 50 Hz frequency

    pwm.start(7.5)           # Neutral position

    return pwm



# Move servo to a specific angle

def move_servo(pwm, angle):

    duty = 2 + (angle / 18)

    pwm.ChangeDutyCycle(duty)

    time.sleep(0.5)
    
    pwm.ChangeDutyCycle(0)  # Stop sending the signal




# Pick up waste

def pick():

    move_servo(base_pwm, 30)

    time.sleep(1)

    move_servo(shoulder_pwm, 165)

    time.sleep(1)

    move_servo(gripper_pwm, 180)

    time.sleep(1)

    move_servo(shoulder_pwm, 30)

    time.sleep(1)



# Release waste

def release():

    move_servo(shoulder_pwm, 165)

    time.sleep(1)

    move_servo(gripper_pwm, 30)

    time.sleep(1)

    move_servo(base_pwm, 30)

    time.sleep(1)

    move_servo(shoulder_pwm, 165)

    time.sleep(1)



# Preprocess image for classification

def preprocess_image(image, target_size=(32, 32)):

    image = cv2.resize(image, target_size)

    image = img_to_array(image) / 255.0

    return np.expand_dims(image, axis=0)



# Classify waste

def classify_waste():

    """

    Capture multiple frames and classify the waste.

    Displays detections and confidences for each frame.

    Retries classification if all frames have confidence below 0.75.

    Returns the class label and confidence if successful.

    """

    while True:  # Loop until confident classification is achieved

        cap = cv2.VideoCapture(0)  # Initialize webcam

        if not cap.isOpened():

            raise Exception("Error accessing the camera")



        print("Capturing frames for classification...")

        frame_count = 35  # Number of frames to capture

        predictions = []



        for frame_num in range(frame_count):

            ret, frame = cap.read()

            if not ret:

                print(f"Frame {frame_num + 1}: Failed to capture, skipping...")

                continue



            # Preprocess frame and predict

            preprocessed_frame = preprocess_image(frame)

            prediction = model.predict(preprocessed_frame)

            class_index = np.argmax(prediction[0])

            confidence = prediction[0][class_index]

            class_label = class_names[class_index]



            # Display each frame's detection and confidence

            print(f"Frame {frame_num + 1}: Detected {class_label} with confidence {confidence:.2f}")



            if confidence > 0.75:

                predictions.append(prediction[0])  # Collect prediction probabilities



            # Allow a slight delay between frames

            time.sleep(0.1)



        cap.release()



        if not predictions:

            print("All captured frames have confidence below 0.75. Retrying classification...")

            continue  # Retry classification



        # Average predictions and determine the most confident class

        avg_prediction = np.mean(predictions, axis=0)

        class_index = np.argmax(avg_prediction)

        class_label = class_names[class_index]

        confidence = avg_prediction[class_index]



        print(f"Final classification: {class_label} with confidence {confidence:.2f}")

        return class_label, confidence



# Move motors forward

def move_forward(speed, duration):

    GPIO.output(IN1, GPIO.LOW)

    GPIO.output(IN2, GPIO.HIGH)

    GPIO.output(IN3, GPIO.LOW)

    GPIO.output(IN4, GPIO.HIGH)

    pwmA.ChangeDutyCycle(speed)

    pwmB.ChangeDutyCycle(speed)

    time.sleep(duration)

    pwmA.ChangeDutyCycle(0)

    pwmB.ChangeDutyCycle(0)



# Move motors backward

def move_backward(speed, duration):

    GPIO.output(IN1, GPIO.HIGH)

    GPIO.output(IN2, GPIO.LOW)

    GPIO.output(IN3, GPIO.HIGH)

    GPIO.output(IN4, GPIO.LOW)

    pwmA.ChangeDutyCycle(speed)

    pwmB.ChangeDutyCycle(speed)

    time.sleep(duration)

    pwmA.ChangeDutyCycle(0)

    pwmB.ChangeDutyCycle(0)



# Setup GPIO and PWM

GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)



# Setup servos

base_pwm = setup_servo(BASE_PIN)

shoulder_pwm = setup_servo(SHOULDER_PIN)

elbow_pwm = setup_servo(ELBOW_PIN)

wrist_pwm = setup_servo(WRIST_PIN)

gripper_pwm = setup_servo(GRIPPER_PIN)



# Setup motors

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwmA = GPIO.PWM(ENA, 100)

pwmB = GPIO.PWM(ENB, 100)

pwmA.start(0)

pwmB.start(0)



try:

    print("Starting waste sorting robot...")



    while True:  # Loop until confident classification is achieved

        # Step 1: Classify waste

        waste_type, confidence = classify_waste()

        print(f"Detected: {waste_type} with confidence {confidence:.2f}")



        # Check confidence threshold

        if confidence > 0.75:

            break  # Exit loop if classification is confident

        else:

            print(f"Confidence {confidence:.2f} is below the threshold. Retrying classification...")



    # Step 2: Pick waste

    print("Picking waste...")

    pick()



    # Step 3: Move to bin

    print(f"Moving to bin for {waste_type}...")

    duration = BIN_DURATIONS[waste_type]

    move_forward(60, duration)



    # Step 4: Release waste

    print("Releasing waste...")

    release()



    # Step 5: Return to original position

    print("Returning to original position...")

    move_backward(60, duration)



except KeyboardInterrupt:

    print("Operation interrupted by user.")



finally:

    # Cleanup

    base_pwm.stop()

    shoulder_pwm.stop()

    elbow_pwm.stop()

    wrist_pwm.stop()

    gripper_pwm.stop()

    pwmA.stop()

    pwmB.stop()

    GPIO.cleanup()

    print("Robot stopped. GPIO cleaned up.")