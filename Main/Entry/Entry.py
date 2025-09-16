import threading
import os 
import pandas as pd
import cv2
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from datetime import datetime, timedelta
# from Main.ServoMotor.ServoMotor import ServoMotor
# from Main.StepperMotor.StepperMotor import StepperMotor

class Entry:
    relativePath = os.getcwd()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logDirectoryPath = os.path.join(base_dir, 'WasteClassificationAppLog')
    # columnNames = {
    #     'Image': pd.Series([], dtype='str'),
    #     'CreatedDate': pd.Series([], dtype='datetime64[ns]'),
    #     'Classification': pd.Series([], dtype='str')
    # }
    createdDateTime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # classificationSummary = pd.DataFrame(columnNames)

    def __init__(self, servoMotorPin, stepperMotorPin, logMessage, cap):
        self.servoMotorPin = servoMotorPin
        self.steppingMotorPin = stepperMotorPin
        # self.servoMotor = ServoMotor(gpio=int(servoMotorPin))
        # self.steppingMotor = StepperMotor(
        #     int(self.steppingMotorPin["IN1"]), 
        #     int(self.steppingMotorPin["IN2"]), 
        #     int(self.steppingMotorPin["IN3"]), 
        #     int(self.steppingMotorPin["IN4"])
        # )
        self.logMessage = logMessage
        self.cap = cap

        recognizeThread = threading.Thread(
            target=self.start_recognize,
            daemon=True
        )
        recognizeThread.start()

    def start_recognize(self):

        device = torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")

        model = resnet18(weights=None)
        num_classes = 4
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load("waste_classifier.pth", map_location=device))
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        classes = ["other", "paper", "plastic", "steel"]
        ret, prev_frame = self.cap.read()
        if not ret:
            self.logMessage("[ERROR] Camera not available for recognition")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        last_detect_time = 0
        cooldown = 1.5 

        self.logMessage("[INFO] Recognition started...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logMessage("[ERROR] Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            non_zero_count = cv2.countNonZero(thresh)
            self.processing = False
            if non_zero_count > 5000 and (datetime.now().timestamp() - last_detect_time > cooldown) and not self.processing:
                self.processing = True
                last_detect_time = datetime.now().timestamp()

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tensor = transform(img_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    label = classes[predicted.item()]

                self.logMessage(f"[INFO] Classified as: {label}")
                self.create_log(label)

                # self.stepperMotor = StepperMotor(
                #     int(self.steppingMotorPin["IN1"]), 
                #     int(self.steppingMotorPin["IN2"]), 
                #     int(self.steppingMotorPin["IN3"]), 
                #     int(self.steppingMotorPin["IN4"]),
                #     test_material = label
                # )
                # time.sleep(1)


                # self.servoMotor = ServoMotor(int(self.servoMotorPin))
                # self.servoMotor.open()
                # time.sleep(1)
                # self.servoMotor.close()
                # time.sleep(1)

                # self.stepperMotor.back_origin()

                self.processing = False 


            prev_gray = gray




    def create_log(self, input):
        currentDate = datetime.now().strftime('%Y-%m-%d')
        if not os.path.isdir(self.logDirectoryPath):
            os.makedirs(self.logDirectoryPath)
        if not os.path.isdir(self.logDirectoryPath + '/' + currentDate):
            os.makedirs(self.logDirectoryPath + '/' + currentDate)

        dailyLog = self.logDirectoryPath + '/' + currentDate + '/' + currentDate + '.log'
        logString = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(input) + '\n'

        with open(dailyLog, 'a+') as file:
            file.write(logString)