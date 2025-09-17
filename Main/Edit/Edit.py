import json
import customtkinter
from PIL import Image
from customtkinter import CTkImage
from Main.Shared.ConfigReader import ConfigReader
from Main.Component.PopUp import PopUp


class Edit:

    def __init__(self, root):
        self.configReader = ConfigReader()
        self.appSettings = self.configReader.read_all_settings()
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
            text='<<',
            font=('Work Sans', 14),
            width=100,
            height=32,
            command=self.next_button_on_click
        )
        self.nextButton.pack(side='right', padx=(0, 10))

        self.connectButton = customtkinter.CTkButton(
            contentFrameNav,
            text='Edit',
            font=('Work Sans', 14),
            width=100,
            height=32,
            command=self.edit_button_on_click
        )
        self.connectButton.pack(side='right', padx=(0, 10))

        self.settingsBox = customtkinter.CTkTextbox(
            self.contentFrameMaster,
            height=200,
            width=400
        )
        self.settingsBox.pack(
            side='top',
            fill='both',
            expand=True,
            padx=20,
            pady=(10, 0)
        )
        self.load_settings_into_textbox()

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

    def log_message(self, text: str):
        self.messageBox.configure(state="normal")
        self.messageBox.insert("end", text + "\n")
        self.messageBox.see("end")
        self.messageBox.configure(state="disabled")

    def load_settings_into_textbox(self):
        self.settingsBox.delete("1.0", "end")
        json_text = json.dumps(self.appSettings, indent=4)
        self.settingsBox.configure(font=("Consolas", 16))
        self.settingsBox.insert("end", json_text)


    def next_button_on_click(self):
        self.destroy()
        from Main.Connect.Connect import Connect
        Connect(self.root)

    def edit_button_on_click(self):
        text = self.settingsBox.get("1.0", "end").strip()
        try:
            safe_text = text.replace("\\", "/")

            newSettings = json.loads(safe_text)

            self.appSettings = newSettings
                
            if self.configReader.update_all_settings(newSettings):
                self.log_message("Settings updated successfully.")
                self.popUp.show_popup(
                    title="Success!",
                    message="Settings updated successfully",
                    isSuccess=True
                )
                self.next_button_on_click()

        except json.JSONDecodeError as e:
            self.log_message(f"Invalid JSON format: {e}")
            self.log_message("Unable to update!")
            self.popUp.show_popup(
                title="Fail!",
                message="Settings did not update successfully",
                isSuccess=False
            )



    def destroy(self):
        self.contentFrameMaster.destroy()
