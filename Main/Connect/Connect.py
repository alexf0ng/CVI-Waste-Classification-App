import cv2
import customtkinter
import threading
from PIL import Image
from customtkinter import CTkImage
from Main.Shared.ConfigReader import ConfigReader
from Main.Component.PopUp import PopUp
from Main.Entry.Entry import Entry


class Connect:
    configReader = ConfigReader()

    def __init__(self, root):
        self.root = root
        self.cap = None
        self.popUp = PopUp(root)

        self.contentFrameMaster = customtkinter.CTkFrame(root)
        self.contentFrameMaster.pack(
            side='right',
            expand=True,
            fill='both',
            padx=10,
            pady=10
        )

        contentFrameNav = customtkinter.CTkFrame(self.contentFrameMaster, fg_color='transparent')
        contentFrameNav.pack(
            side='top',
            fill='x',
            padx=20,
            pady=(30, 0)
        )

        label = customtkinter.CTkLabel(contentFrameNav, text='Read', font=('Work Sans', 17))
        label.pack(side='left')

        self.nextButton = customtkinter.CTkButton(
            contentFrameNav,
            text='>>',
            font=('Work Sans', 14),
            width=100,
            height=32,
            command=self.next_button_on_click
        )
        self.nextButton.pack(side='right', padx=(0, 10))

        self.connectButton = customtkinter.CTkButton(
            contentFrameNav,
            text='Connect',
            font=('Work Sans', 14),
            width=100,
            height=32,
            command=self.connect_button_on_click
        )
        self.connectButton.pack(side='right', padx=(0, 10))

        self.disconnectButton = customtkinter.CTkButton(
            contentFrameNav,
            text='Disconnect',
            font=('Work Sans', 14),
            width=100,
            height=32,
            command=self.disconnect_button_on_click
        )
        self.disconnectButton.pack_forget()

        self.cameraFrame = customtkinter.CTkFrame(self.contentFrameMaster, fg_color="black")
        self.cameraFrame.pack(
            side='top',
            expand=True,
            fill='both',
            padx=20,
            pady=(10, 0)
        )

        self.video_label = customtkinter.CTkLabel(
            self.cameraFrame,
            text="Not connected",
            font=("Work Sans", 16)
        )
        self.video_label.pack(expand=True)

        self.messageBox = customtkinter.CTkTextbox(
            self.contentFrameMaster,
            height=100,
            width=400
        )
        self.messageBox.pack(
            side='top',
            fill='x',
            padx=20,
            pady=(10, 0)
        )
        self.messageBox.insert("end", "Status messages will appear here...\n")
        self.messageBox.configure(state="disabled")

    def delay_disconnect_button(self):
        self.disconnectButton.pack(side='right')

    def log_message(self, text: str):
        self.messageBox.configure(state="normal")
        self.messageBox.insert("end", text + "\n")
        self.messageBox.see("end")
        self.messageBox.configure(state="disabled")

    def next_button_on_click(self):
        self.destroy()
        from Main.Edit.Edit import Edit
        Edit(self.root)

    def connect_button_on_click(self):
        self.connectButton.pack_forget()
        self.nextButton.pack_forget()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                self.log_message("Failed to open camera!")
                self.video_label.configure(text="Not connected")
                self.popUp.close_loading_popup()
                return

            self.log_message("Camera connected.")
            self.update_frame()

            self.root.after(10, self.entry)
            self.disconnectButton.after(6000, self.delay_disconnect_button)

        else:
            self.log_message("Camera already connected.")

    def disconnect_button_on_click(self):
        self.entryThread.join(2)
        self.disconnectButton.pack_forget()
        self.nextButton.pack(
            side='right',
            padx=(0, 10)
        )
        self.connectButton.pack(
            side='right',
            padx=(0, 10)
        )

    def entry(self):      
        appSettings = self.configReader.read_all_settings()
        self.entryThread = threading.Thread(
            target=self.entry_start,
            args=(appSettings,),
            daemon=True,

        )
        self.entryThread.start()


    def entry_start(self, appSettings):
        self.entry = Entry(appSettings['ServoMotor'][0]['Signal'], appSettings['SteppingMotor'][0], self.log_message, self.cap, appSettings['RaspberryPi'], appSettings['ModelPath'], appSettings['BackgroundPath'])
        

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(50, self.update_frame)
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            imgtk = CTkImage(light_image=img, size=(800, 500))
            self.video_label.configure(image=imgtk, text="")
            self.video_label.imgtk = imgtk

            self.root.after(10, self.update_frame)
            
    

    def destroy(self):
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.contentFrameMaster.destroy()
