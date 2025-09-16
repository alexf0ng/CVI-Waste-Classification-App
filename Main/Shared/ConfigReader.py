import os
import json
import sys

class ConfigReader:
    def __init__(self):
        self.template = {
            "ServoMotor": [
                {
                    "Signal": "17"
                }
            ],
            "SteppingMotor" : [
                {
                    "IN1" : "17",
                    "IN2" : "18", 
                    "IN3" : "27",
                    "IN4" : "22",
                }
            ]
        }

        self.env = "Demo"

        self.relativePath = os.getcwd()

        self.fileName = "AppSettings." + self.env.upper() + ".json"

        self.initialize_app_settings()


    
    def initialize_app_settings(self):
        try:
            if not os.path.isdir(self.relativePath):
                os.makedirs(self.relativePath)
        
            appSettingsDir = os.path.join(self.relativePath, "AppSettings" + self.env.upper())

            if not os.path.isdir(appSettingsDir):
                os.makedirs(appSettingsDir)
            
            appSettingsFile = os.path.join(appSettingsDir, 'AppSettings.' + self.env.upper() + '.json')

            if os.path.isfile(appSettingsFile) and os.path.getsize(appSettingsFile) > 0:
                pass
            else:
                with open(appSettingsFile, 'w') as file:
                    json.dump(self.template, file, indent=4)
            
            self.filePath = appSettingsFile
    
        except Exception:
            basePath = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            self.filePath = os.path.join(basePath, 'AppSettings.json')

    def read_all_settings(self):
        with open(self.filePath, "r") as file:
            config = json.load(file)

        if config is None:
            return None
        
        return config
    
    def update_all_settings(self, newSettings):
        try:
            with open(self.filePath, "w") as file:
                json.dump(newSettings, file, indent=4)

            return True
        except Exception as e:
            return False