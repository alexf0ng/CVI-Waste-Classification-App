
import customtkinter
import sys
from Main.Connect.Connect import Connect

class Home:
    def __init__(self, root):
        self.root = root
        root.title('Waste Classifier App')

        root.grid_rowconfigure(0, weight = 1)
        root.grid_columnconfigure(0, weight = 1)

        self.mainFrame = customtkinter.CTkFrame(root)
        self.mainFrame.pack(
            fill = 'both',
            expand = True,
            padx = 10,
            pady = 10
        )

        self.contentFrame = Connect(self.mainFrame)

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    
    def on_close(self):
        self.root.quit()
        self.root.destroy()
        sys.exit(0)



if __name__ == "__main__":
    root = customtkinter.CTk()
    root.geometry("800x600")

    app = Home(root)
    root.mainloop()
