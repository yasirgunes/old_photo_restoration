import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
from utils.pipeline import pipeline  # or from pipeline import pipeline depending on your structure

class PhotoRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Old Photo Restoration")
        self.root.geometry("1200x800")  # Made taller to accommodate status list
        
        # Create frames
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        
        # Create browse button
        self.browse_button = tk.Button(
            self.button_frame, 
            text="Browse Image", 
            command=self.browse_image
        )
        self.browse_button.pack(pady=5)
        
        # Create status list frame
        self.status_frame = tk.Frame(root)
        self.status_frame.pack(pady=5)
        
        # Create status labels
        self.status_labels = []
        self.status_steps = [
            "1. Removing Scratches",
            "2. Implementing noise reduction",
            "3. Histogram Equalization",
            "4. Adjusting sharpness",
            "5. Adjusting brightness and contrast",
            "6. Complete Restoration"
        ]
        
        for step in self.status_steps:
            label = tk.Label(self.status_frame, text=f"{step} (Pending...)", font=('Arial', 10))
            label.pack(anchor='w', padx=20)
            self.status_labels.append(label)
        
        # Create image frame
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
        
        # Initialize current step
        self.current_step = 0
        
    def update_status(self, status):
        if "scratches" in status.lower():
            self.update_step_status(0, "Done!")
        elif "noise" in status.lower():
            self.update_step_status(1, "Done!")
        elif "histogram" in status.lower():
            self.update_step_status(2, "Done!")
        elif "sharpness" in status.lower():
            self.update_step_status(3, "Done!")
        elif "brightness" in status.lower():
            self.update_step_status(4, "Done!")
        
        self.root.update_idletasks()
    
    def update_step_status(self, step_index, status):
        # Reset all subsequent steps to "Pending..."
        for i in range(step_index + 1, len(self.status_labels)):
            self.status_labels[i].config(
                text=f"{self.status_steps[i]} (Pending...)",
                fg='black'
            )
        
        # Update current step
        self.status_labels[step_index].config(
            text=f"{self.status_steps[step_index]} ({status})",
            fg='green' if status == "Done!" else 'black'
        )
    
    def browse_image(self):
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg')
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            # Reset status labels
            for i, label in enumerate(self.status_labels):
                label.config(text=f"{self.status_steps[i]} (Pending...)", fg='black')
            
            # Disable browse button during processing
            self.browse_button.config(state='disabled')
            
            try:
                # Read image
                original = cv2.imread(file_path)
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                
                # Display original image immediately
                self.display_image(original, self.image_label1)
                
                # Start processing in a separate thread
                thread = threading.Thread(
                    target=self.process_image,
                    args=(original,)
                )
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                self.browse_button.config(state='normal')
    
    def process_image(self, original):
        try:
            # Process image
            restored = pipeline(original, status_callback=self.update_status)
            
            # Update GUI in the main thread
            self.root.after(0, self.display_restored_image, restored)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
        finally:
            # Re-enable browse button
            self.root.after(0, lambda: self.browse_button.config(state='normal'))
    
    def display_image(self, img, label):
        # Convert to PIL format
        pil_img = Image.fromarray(img)
        
        # Resize image
        max_size = (500, 500)
        pil_img.thumbnail(max_size)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Update label
        label.configure(image=photo)
        label.image = photo  # Keep a reference!
    
    def display_restored_image(self, restored):
        self.display_image(restored, self.image_label2)
        # Update final status
        self.update_step_status(5, "Done!")

def main():
    root = tk.Tk()
    app = PhotoRestorationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()