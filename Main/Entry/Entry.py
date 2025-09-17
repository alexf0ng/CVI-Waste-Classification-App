import threading
import os 
import pandas as pd
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from PIL import Image
from torchvision.models import resnet18
from datetime import datetime, timedelta
from Main.ServoMotor.ServoMotor import ServoMotor
from Main.StepperMotor.StepperMotor import StepperMotor

class Entry:
    relativePath = os.getcwd()
    logDirectoryPath = os.path.join(relativePath, 'WasteClassificationAppLog')

    columnNames = {
        'Image': pd.Series([], dtype='str'),
        'Object Mask': pd.Series([], dtype='str'),
        'Segmented Object': pd.Series([], dtype='str'),
        'Dominant Color': pd.Series([], dtype='str'),
        'Dominant Color(RGB)': pd.Series([], dtype='str'),
        'CreatedDate': pd.Series([], dtype='datetime64[ns]'),
        'Classification': pd.Series([], dtype='str')
    }
    createdDateTime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    classificationSummary = pd.DataFrame(columnNames)

    def __init__(self, servoMotorPin, stepperMotorPin, logMessage, cap, raspberryPi, modelPath, backgroundPath):
        self.servoMotorPin = servoMotorPin
        self.steppingMotorPin = stepperMotorPin
        self.raspberryPi = raspberryPi
        self.modelPath = modelPath
        self.backgroundPath = backgroundPath

        self.logMessage = logMessage
        self.cap = cap
        self.kill = threading.Event()
        self.recognizeThread = None

    def start_recognize_thread(self):
        recognizeThread = threading.Thread(
            target=self.start_recognize,
            daemon=True
        )
        recognizeThread.start()
    
    def stop(self):
        self.kill.set()
        if self.recognizeThread is not None:
            self.recognizeThread.join(2)

    # def start_recognize(self):

    #     device = torch.device("mps" if torch.backends.mps.is_available() else
    #                         "cuda" if torch.cuda.is_available() else "cpu")
        
    #     try:
    #         model = resnet18(weights=None)
    #         num_classes = 4
    #         model.fc = nn.Linear(model.fc.in_features, num_classes)
    #         model.load_state_dict(torch.load(self.modelPath, map_location=device))
    #         model = model.to(device)
    #         model.eval()

    #         transform = transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406],
    #                                 [0.229, 0.224, 0.225])
    #         ])

    #         classes = ["other", "paper", "plastic", "steel"]
    #         ret, prev_frame = self.cap.read()
    #         if not ret:
    #             self.logMessage("[ERROR] Camera not available for recognition")
    #             return

    #         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    #         last_detect_time = 0
    #         cooldown = 1.5 

    #         self.logMessage("[INFO] Recognition started...")

    #         while not self.kill.is_set():
    #             time.sleep(3)
    #             ret, frame = self.cap.read()
    #             if not ret:
    #                 self.logMessage("[ERROR] Failed to grab frame")
    #                 break

    #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             diff = cv2.absdiff(prev_gray, gray)
    #             _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    #             non_zero_count = cv2.countNonZero(thresh)
    #             self.processing = False
    #             if non_zero_count > 5000 and (datetime.now().timestamp() - last_detect_time > cooldown) and not self.processing:
    #                 self.processing = True
    #                 last_detect_time = datetime.now().timestamp()

    #                 newFrame, objectMask, segmentedObject, colorPatch, dominantColor = self.segmentation(frame, self.backgroundPath)
    #                 imagePath = self.save_image(newFrame, objectMask, segmentedObject, colorPatch)

    #                 img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 img_pil = Image.fromarray(img_rgb)
    #                 img_tensor = transform(img_pil).unsqueeze(0).to(device)

    #                 with torch.no_grad():
    #                     outputs = model(img_tensor)
    #                     _, predicted = torch.max(outputs, 1)
    #                     label = classes[predicted.item()]

    #                 self.logMessage(f"[INFO] Classified as: {label}")
    #                 self.create_log(label)

    #                 if self.raspberryPi:
    #                     self.stepperMotor = StepperMotor(
    #                         int(self.steppingMotorPin["IN1"]), 
    #                         int(self.steppingMotorPin["IN2"]), 
    #                         int(self.steppingMotorPin["IN3"]), 
    #                         int(self.steppingMotorPin["IN4"]),
    #                         test_material = label
    #                     )
    #                     time.sleep(2)

    #                     self.servoMotor = ServoMotor(int(self.servoMotorPin))
    #                     time.sleep(1)

    #                     self.servoMotor.open()
    #                     time.sleep(1)

    #                     self.servoMotor.close()
    #                     time.sleep(1)
                        
    #                     self.stepperMotor.back_origin()
    #                     time.sleep(2)

    #                 self.create_task_history(imagePath, dominantColor, label)

    #                 self.processing = False 
    #                 time.sleep(2)


    #             prev_gray = gray

    #     except Exception as e:
    #         self.logMessage(f"Fail due to {str(e)}")
    #         self.create_log(f"Fail due to str{e}")
    def start_recognize(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")

        try:
            model = resnet18(weights=None)
            num_classes = 4
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(self.modelPath, map_location=device))
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
            
            self.processing = False
            self.is_moving = False
            
            self.logMessage("[INFO] Recognition started...")

            while not self.kill.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    self.logMessage("[ERROR] Failed to grab frame")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                non_zero_count = cv2.countNonZero(thresh)
                
                if non_zero_count > 5000 and not self.is_moving and not self.processing:
                    self.logMessage("[INFO] Motion started...")
                    self.is_moving = True
                    self.processing = True

                if self.is_moving:
                    if non_zero_count < 1000 and (datetime.now().timestamp() - last_detect_time > cooldown):
                        self.is_moving = False
                        last_detect_time = datetime.now().timestamp()
                        self.logMessage("[INFO] Motion stopped. Capturing and classifying...")

                        current_frame_for_analysis = frame 
                        
                        newFrame, objectMask, segmentedObject, colorPatch, dominantColor = self.segmentation(current_frame_for_analysis, self.backgroundPath)
                        imagePath = self.save_image(newFrame, objectMask, segmentedObject, colorPatch)

                        img_rgb = cv2.cvtColor(current_frame_for_analysis, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        img_tensor = transform(img_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(img_tensor)
                            _, predicted = torch.max(outputs, 1)
                            label = classes[predicted.item()]

                        self.logMessage(f"[INFO] Classified as: {label}")
                        self.create_log(label)
                        
                        if self.raspberryPi:
                            self.stepperMotor = StepperMotor(
                                int(self.steppingMotorPin["IN1"]), 
                                int(self.steppingMotorPin["IN2"]), 
                                int(self.steppingMotorPin["IN3"]), 
                                int(self.steppingMotorPin["IN4"]),
                                raspberryPi = self.raspberryPi,
                                test_material = label
                            )
                            time.sleep(2)
                            self.servoMotor = ServoMotor(int(self.servoMotorPin), self.raspberryPi)
                            time.sleep(1)
                            self.servoMotor.open()
                            time.sleep(1)
                            self.servoMotor.close()
                            time.sleep(1)
                            self.stepperMotor.back_origin()
                            time.sleep(2)
                            
                        self.create_task_history(imagePath, dominantColor, label)
                        
                        self.processing = False
                        time.sleep(1)
                
                prev_gray = gray

        except Exception as e:
            self.logMessage(f"Fail due to {str(e)}")
            self.create_log(f"Fail due to str{e}")


    def save_image(self, frame, objectMask, segmentedObject, colorPatch):
        currentDate = datetime.now().strftime('%Y-%m-%d')
        currentTime = datetime.now().strftime('%H%M%S')

        baseDir = os.path.join(self.logDirectoryPath, currentDate, "Image", currentTime)
        os.makedirs(os.path.join(baseDir, "Original"), exist_ok=True)
        os.makedirs(os.path.join(baseDir, "ObjectMask"), exist_ok=True)
        os.makedirs(os.path.join(baseDir, "SegmentedObject"), exist_ok=True)
        os.makedirs(os.path.join(baseDir, "DominantColor"), exist_ok=True)

        try:
            imagePath = os.path.join(baseDir, "Original", f"{currentTime}.jpg")
            objectMaskPath = os.path.join(baseDir, "ObjectMask", f"{currentTime}.jpg")
            segmentedObjectPath = os.path.join(baseDir, "SegmentedObject", f"{currentTime}.jpg")
            dominantColorPath = os.path.join(baseDir, "DominantColor", f"{currentTime}.jpg")

            cv2.imwrite(imagePath, frame)
            cv2.imwrite(objectMaskPath, objectMask)
            cv2.imwrite(segmentedObjectPath, segmentedObject)
            cv2.imwrite(dominantColorPath, colorPatch)

            return [imagePath, objectMaskPath, segmentedObjectPath, dominantColorPath]

        except Exception as e:
            self.create_log(f"Unable to store the image due to {str(e)}")
            return [str(e)] * 4



    def create_log(self, input):
        currentDate = datetime.now().strftime('%Y-%m-%d')
        if not os.path.isdir(self.logDirectoryPath):
            os.makedirs(self.logDirectoryPath)
        if not os.path.isdir(self.logDirectoryPath + '/' + currentDate):
            os.makedirs(self.logDirectoryPath + '/' + currentDate)
        if not os.path.isdir(self.logDirectoryPath + '/' + currentDate + '/Log'):
            os.makedirs(self.logDirectoryPath + '/' + currentDate + '/Log')

        dailyLog = self.logDirectoryPath + '/' + currentDate + '/Log/' + currentDate + '.log'
        logString = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(input) + '\n'

        with open(dailyLog, 'a+') as file:
            file.write(logString)


    def create_task_history(self, imagePath, dominantColor, label):
        currentDate = datetime.now().strftime('%Y-%m-%d')
        if not os.path.isdir(self.logDirectoryPath):
            os.makedirs(self.logDirectoryPath)
        if not os.path.isdir(self.logDirectoryPath+ '/' + currentDate):
            os.makedirs(self.logDirectoryPath+ '/' + currentDate)
        if not os.path.isdir(self.logDirectoryPath+ '/' + currentDate + '/TaskHistory'):
            os.makedirs(self.logDirectoryPath+ '/' + currentDate + '/TaskHistory')

        try:
            newRow = {
                'Image': imagePath[0],
                'Object Mask': imagePath[1],
                'Segmented Object': imagePath[2],
                'Dominant Color': imagePath[3],
                'Dominant Color(RGB)': dominantColor,
                'CreatedDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Classification': label,
            }
            self.classificationSummary = pd.concat([self.classificationSummary, pd.DataFrame([newRow])], ignore_index=True)
            self.classificationSummary.to_csv(self.logDirectoryPath+ '/' + currentDate + '/TaskHistory/' + self.createdDateTime +'ClassificationSummary.csv', index=False)
        except Exception as e:
            self.create_log('Cannot update ClassificationSummary.csv file, close the csv file so that it can be updated')

    def segmentation(self, frame, backgroundImagePath):
        backgroundFrame = cv2.imread(backgroundImagePath)

        if frame is None or backgroundFrame is None:
            self.create_log("Error: Could not load one of the images. Check the file paths.")
            return frame, None, None, None, None
    
        objectMask = self.get_object_mask(frame, backgroundFrame)

        if np.sum(objectMask) < 1000:
            self.create_log("No object detected. The mask is empty.")
            return frame, None, None, None, None

        segmentedObject = cv2.bitwise_and(frame, frame, mask=objectMask)
        dominantColor = self.get_dominant_color(cv2.cvtColor(segmentedObject, cv2.COLOR_BGR2RGB), objectMask)
        # print("Dominant color (R,G,B):", dominantColor)

        colorPatch = np.zeros((100, 100, 3), dtype=np.uint8)
        colorPatch[:] = dominantColor[::-1]

        return frame, objectMask, segmentedObject, colorPatch, dominantColor

    def get_dominant_color(self, image, mask, k=3):
        pixels = image[mask == 255]
        if len(pixels) == 0:
            return (0, 0, 0)

        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)

        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]

        return tuple(map(int, dominant))
    
    def get_object_mask(self, frame, background_frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_frame, gray_bg)
        
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask