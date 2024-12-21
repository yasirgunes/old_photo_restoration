import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading

class PhotoRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Old Photo Restoration")
        self.root.minsize(800, 600)
        self.root.configure(bg='#f0f2f5')
        
        # Create scrollable main container
        self.canvas = tk.Canvas(root, bg='#f0f2f5')
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f0f2f5')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True, padx=20)
        self.scrollbar.pack(side="right", fill="y")
        
        # Title
        self.title_label = tk.Label(
            self.scrollable_frame,
            text="Old Photo Restoration",
            font=('Helvetica', 24, 'bold'),
            bg='#f0f2f5'
        )
        self.title_label.pack(pady=(20, 10))
        
        # Browse button
        self.browse_button = tk.Button(
            self.scrollable_frame,
            text="Browse Image",
            command=self.browse_image,
            width=15,
            font=('Helvetica', 10),
            relief='solid',
            bg='white'
        )
        self.browse_button.pack(pady=(0, 20))
        
        # Status frame
        self.status_frame = tk.Frame(self.scrollable_frame, bg='white', relief='solid', bd=1)
        self.status_frame.pack(padx=50, pady=(0, 20), fill='x')
        
        # Status title
        self.status_title = tk.Label(
            self.status_frame,
            text="Restoration Progress",
            font=('Helvetica', 12, 'bold'),
            bg='white',
            pady=10
        )
        self.status_title.pack()
        
        # Status labels
        self.status_labels = []
        self.status_steps = [
            "Removing Scratches",
            "Implementing noise reduction",
            "Histogram Equalization",
            "Adjusting sharpness",
            "Adjusting brightness and contrast",
            "Complete Restoration"
        ]
        
        for i, step in enumerate(self.status_steps, 1):
            label = tk.Label(
                self.status_frame,
                text=f"{i}. {step} (Pending...)",
                font=('Helvetica', 10),
                bg='white',
                pady=5,
                padx=20,
                anchor='w'
            )
            label.pack(fill='x')
            self.status_labels.append(label)
        
        # Image frame
        self.image_frame = tk.Frame(self.scrollable_frame, bg='#f0f2f5')
        self.image_frame.pack(pady=20)
        
        # Create containers for images
        self.original_container = tk.Frame(self.image_frame, bg='white', bd=1, relief='solid')
        self.restored_container = tk.Frame(self.image_frame, bg='white', bd=1, relief='solid')
        
        # Configure grid with proper spacing
        self.image_frame.grid_columnconfigure(1, minsize=20)
        self.original_container.grid(row=0, column=0, padx=10)
        self.restored_container.grid(row=0, column=2, padx=10)
        
        # Image labels
        tk.Label(self.original_container, text="Original Image", font=('Helvetica', 10, 'bold'), bg='white').pack(pady=5)
        tk.Label(self.restored_container, text="Restored Image", font=('Helvetica', 10, 'bold'), bg='white').pack(pady=5)
        
        self.image_label1 = tk.Label(self.original_container, bg='white')
        self.image_label2 = tk.Label(self.restored_container, bg='white')
        self.image_label1.pack(padx=10, pady=10)
        self.image_label2.pack(padx=10, pady=10)
        
        # Create initial placeholder images
        self.create_placeholder_images()
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)
        self.current_width = self.root.winfo_width()

    def on_window_resize(self, event):
        if event.widget == self.root and event.width != self.current_width:
            self.current_width = event.width
            if hasattr(self, 'current_original'):
                self.display_image(self.current_original, self.image_label1)
            if hasattr(self, 'current_restored'):
                self.display_image(self.current_restored, self.image_label2)

    def create_placeholder_images(self):
        placeholder = Image.new('RGB', (300, 300), '#f5f5f5')
        photo = ImageTk.PhotoImage(placeholder)
        
        self.image_label1.configure(image=photo)
        self.image_label1.image = photo
        
        self.image_label2.configure(image=photo)
        self.image_label2.image = photo

    def update_status(self, status):
        step_index = -1
        if "scratches" in status.lower():
            step_index = 0
        elif "noise" in status.lower():
            step_index = 1
        elif "histogram" in status.lower():
            step_index = 2
        elif "sharpness" in status.lower():
            step_index = 3
        elif "brightness" in status.lower():
            step_index = 4
            
        if step_index != -1:
            self.root.after(0, lambda: self.update_step_status(step_index, "Done!"))

    def update_step_status(self, step_index, status):
        for i in range(step_index + 1, len(self.status_labels)):
            self.status_labels[i].config(
                text=f"{i + 1}. {self.status_steps[i]} (Pending...)",
                fg='#666666'
            )
        
        self.status_labels[step_index].config(
            text=f"{step_index + 1}. {self.status_steps[step_index]} ({status})",
            fg='#4CAF50' if status == "Done!" else '#666666'
        )

    def browse_image(self):
        file_types = [('Image files', '*.png *.jpg *.jpeg')]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            # Reset status labels and clear restored image
            for i, label in enumerate(self.status_labels):
                label.config(
                    text=f"{i + 1}. {self.status_steps[i]} (Pending...)",
                    fg='#666666'
                )
            
            # Clear the restored image and show placeholder
            self.create_placeholder_images()
            
            # Disable browse button and show processing state
            self.browse_button.config(state='disabled', text="Processing...", bg='#9e9e9e')
            
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
                self.browse_button.config(
                    state='normal',
                    text="Browse Image",
                    bg='white'
                )

    def process_image(self, original):
        try:
            from utils.pipeline import pipeline  # Import here to avoid circular imports
            restored = pipeline(original, status_callback=self.update_status)
            self.root.after(0, self.display_restored_image, restored)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.browse_button.config(
                state='normal',
                text="Browse Image",
                bg='white'
            ))

    def display_image(self, img, label):
        # Store current image for resize handling
        if label == self.image_label1:
            self.current_original = img
        else:
            self.current_restored = img
            
        # Calculate size based on window width
        window_width = self.root.winfo_width()
        max_img_width = min(300, (window_width - 100) // 2)
        max_img_height = 400
        
        # Convert to PIL format
        pil_img = Image.fromarray(img)
        
        # Calculate aspect ratio
        aspect_ratio = pil_img.width / pil_img.height
        
        # Determine new size maintaining aspect ratio
        if aspect_ratio > 1:
            new_width = max_img_width
            new_height = int(new_width / aspect_ratio)
            if new_height > max_img_height:
                new_height = max_img_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_height = max_img_height
            new_width = int(new_height * aspect_ratio)
            if new_width > max_img_width:
                new_width = max_img_width
                new_height = int(new_width / aspect_ratio)
        
        # Resize image
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and update label
        photo = ImageTk.PhotoImage(pil_img)
        label.configure(image=photo)
        label.image = photo

    def display_restored_image(self, restored):
        self.display_image(restored, self.image_label2)
        self.update_step_status(5, "Done!")

def main():
    root = tk.Tk()
    app = PhotoRestorationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()