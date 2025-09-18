import os
import json
import sys
import platform

class ConfigReader:
    def __init__(self):
        system = platform.system()
        relative = os.getcwd()

        if system == "Linux":
            self.template = {
                "ServoMotor": [
                    {
                        "Signal": "13"
                    }
                ],
                "SteppingMotor" : [
                    {
                        "IN1" : "17",
                        "IN2" : "18", 
                        "IN3" : "27",
                        "IN4" : "22",
                    }
                ],
                "RaspberryPi" : True,
                "ModelPath" : "/home/alexfong/PythonProjects/waste_classifier.pth",
                "BackgroundPath": "/home/alexfong/PythonProjects/Background.jpg"    
            }
        else:
            self.template = {
                "ServoMotor": [
                    {
                        "Signal": "13"
                    }
                ],
                "SteppingMotor" : [
                    {
                        "IN1" : "17",
                        "IN2" : "18", 
                        "IN3" : "27",
                        "IN4" : "22",
                    }
                ],
                "RaspberryPi" : False,
                "ModelPath" : f"{relative}/_internal/waste_classifier.pth",
                "BackgroundPath": f"{relative}/_internal/Background.jpg"    
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
            # self.update_path()
    
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
        

    # def update_path(self, modelPath=None, bgPath=None):
    #     if getattr(sys, 'frozen', False):
    #         base_dir = sys._MEIPASS 
    #     else:
    #         base_dir = os.getcwd() 

    #     modelPath = modelPath or os.path.join(base_dir, "waste_classifier.pth")
    #     bgPath = bgPath or os.path.join(base_dir, "Background.jpg")

    #     self.template['ModelPath'] = modelPath
    #     self.template['BackgroundPath'] = bgPath

    #     try:
    #         if hasattr(self, 'filePath') and self.filePath:
    #             with open(self.filePath, 'r') as f:
    #                 config = json.load(f)

    #             config['ModelPath'] = modelPath
    #             config['BackgroundPath'] = bgPath

    #             with open(self.filePath, 'w') as f:
    #                 json.dump(config, f, indent=4)
    #     except Exception as e:
    #         print(f"Failed to update paths in config: {e}")
