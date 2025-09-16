import customtkinter

class PopUp:
    def __init__(self, root):
        self.root = root
        self.loading_popup = None  
    
    def show_popup(self, title, message, isSuccess=False):
        popup = customtkinter.CTkToplevel(self.root)
        popup.title(title)
        popup.geometry("300x150")

        warning_icon = customtkinter.CTkLabel(
            popup,
            text='✔' if isSuccess else '❌ ',
            font=('Arial', 35),
            text_color='green' if isSuccess else 'red'
        )
        warning_icon.pack(pady=(20, 10))

        label = customtkinter.CTkLabel(
            popup,
            text=message,
            font=('Arial', 14),
        )
        label.pack(pady=(0, 20))

        close_button = customtkinter.CTkButton(
            popup,
            text="OK",
            command=popup.destroy,
        )
        close_button.pack()

        popup.transient(self.root)
        popup.grab_set()
        popup.wait_window()

    def show_popup_entry(self, title, message):
        popup = customtkinter.CTkToplevel(self.root)
        popup.title(title)
        popup.geometry("400x180")

        result = {'value': None}

        label = customtkinter.CTkLabel(
            popup,
            text=message,
            font=('Arial', 14),
        )
        label.pack(pady=(10, 5))

        entry = customtkinter.CTkEntry(popup, width=300)
        entry.pack(pady=(0, 10))

        def on_ok():
            result['value'] = entry.get()
            popup.destroy()

        def on_close():
            result['value'] = 'CLOSED'  # Special value if window closed
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", on_close)  # Handle 'X' close button

        ok_button = customtkinter.CTkButton(
            popup,
            text="OK",
            command=on_ok
        )
        ok_button.pack()

        popup.transient(self.root)
        popup.grab_set()
        popup.wait_window()
        return result['value']
    
    
    def show_loading_popup(self, message="Loading..."):
        popup = customtkinter.CTkToplevel(self.root)
        popup.title("Please wait")
        popup.geometry("300x120")
        popup.resizable(False, False)

        label = customtkinter.CTkLabel(
            popup,
            text=message,
            font=('Arial', 14),
        )
        label.pack(pady=20)

        progressbar = customtkinter.CTkProgressBar(popup, mode='indeterminate', width=200)
        progressbar.pack(pady=10)
        progressbar.start()

        popup.transient(self.root)
        popup.grab_set()
        popup.update()
        self.loading_popup = popup
    
    def close_loading_popup(self):
        if self.loading_popup is not None:
            self.loading_popup.destroy()
            self.loading_popup = None