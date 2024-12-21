import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from utils.pipeline import pipeline

class PhotoRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Old Photo Restoration")
        self.root.geometry("1200x600")
        
        # Create frames
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)
        
        # Create labels for images
        self.original_label = tk.Label(self.image_frame, text="Original Image")
        self.original_label.grid(row=0, column=0, padx=10)
        
        self.restored_label = tk.Label(self.image_frame, text="Restored Image")
        self.restored_label.grid(row=0, column=1, padx=10)
        
        self.image_label1 = tk.Label(self.image_frame)
        self.image_label1.grid(row=1, column=0, padx=10)
        
        self.image_label2 = tk.Label(self.image_frame)
        self.image_label2.grid(row=1, column=1, padx=10)
        
        # Create browse button
        self.browse_button = tk.Button(
            self.button_frame, 
            text="Browse Image", 
            command=self.browse_image
        )
        self.browse_button.pack(pady=5)
        
        # Create status label
        self.status_label = tk.Label(self.button_frame, text="")
        self.status_label.pack(pady=5)
        
    def browse_image(self):
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg')
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            try:
                # Read and process image
                original = cv2.imread(file_path)
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                
                # Update status
                self.status_label.config(text="Starting restoration...")
                self.root.update_idletasks()
                
                # Fix: Pass status_callback as second argument
                restored = pipeline(original, status_callback=self.update_status)
                
                # Convert images to PIL format for display
                original_pil = Image.fromarray(original)
                restored_pil = Image.fromarray(restored)
                
                # Resize images to fit GUI
                max_size = (500, 500)
                original_pil.thumbnail(max_size)
                restored_pil.thumbnail(max_size)
                
                # Convert to PhotoImage
                self.photo1 = ImageTk.PhotoImage(original_pil)
                self.photo2 = ImageTk.PhotoImage(restored_pil)
                
                # Update labels
                self.image_label1.configure(image=self.photo1)
                self.image_label2.configure(image=self.photo2)
                
                # Clear status
                self.status_label.config(text="")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                
    def update_status(self, status):
        self.status_label.config(text=status)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = PhotoRestorationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()