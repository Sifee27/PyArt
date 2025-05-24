"""
File converter interface for PyArt
Interactive interface for image/video file conversion to ASCII art
"""

import os
import sys
import cv2
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.effects import EffectProcessor
from src.file_processor import FileProcessor
from src.utils import get_timestamp, load_config, save_config


class FileConverterApp:
    """GUI application for converting images and videos to ASCII art"""
    
    def __init__(self, root):
        """Initialize the file converter application"""
        self.root = root
        self.root.title("PyArt - File Converter")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Load config
        self.config = load_config()
        
        # Initialize components
        self.effect_processor = EffectProcessor()
        self.file_processor = FileProcessor(self.effect_processor)
        
        # State variables
        self.current_file = None
        self.current_file_type = None  # 'image' or 'video'
        self.current_effect = 'ascii_blocks'
        self.processing_video = False
        self.original_image = None
        self.processed_image = None
        self.preview_image = None
        self.video_info = None
        
        # Create UI
        self.create_menu()
        self.create_main_ui()
        
        # Set initial effect
        self.effect_var.set(self.current_effect)
        
        # Update supported formats text
        formats = self.file_processor.get_supported_formats()
        img_formats = ", ".join(formats['image'])
        vid_formats = ", ".join(formats['video'])
        self.supported_formats_label.config(
            text=f"Supported Images: {img_formats}\nSupported Videos: {vid_formats}"
        )
    
    def create_menu(self):
        """Create application menu"""
        self.menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image_file)
        file_menu.add_command(label="Open Video", command=self.open_video_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Processed Image", command=self.save_processed_image)
        file_menu.add_command(label="Process Video", command=self.process_full_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        
        # Add menus to menubar
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=self.menu_bar)
    
    def create_main_ui(self):
        """Create main application UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection
        ttk.Label(control_frame, text="File Selection:").pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(
            control_frame, 
            text="Open Image", 
            command=self.open_image_file
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            control_frame, 
            text="Open Video", 
            command=self.open_video_file
        ).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Effect selection
        ttk.Label(control_frame, text="ASCII Effect:").pack(anchor=tk.W, pady=(0, 5))
        
        self.effect_var = tk.StringVar(value=self.current_effect)
        effect_names = self.effect_processor.get_effect_names()
        ascii_effects = [e for e in effect_names if e.startswith('ascii_')]
        
        effect_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.effect_var,
            values=ascii_effects,
            state="readonly"
        )
        effect_dropdown.pack(fill=tk.X, pady=2)
        effect_dropdown.bind("<<ComboboxSelected>>", self.on_effect_changed)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
          # ASCII Detail level
        ttk.Label(control_frame, text="ASCII Detail Level:").pack(anchor=tk.W, pady=(0, 5))
        
        self.detail_var = tk.DoubleVar(value=0.5)  # Lower default for better performance
        detail_scale = ttk.Scale(
            control_frame,
            from_=0.0,
            to=2.0,
            variable=self.detail_var,
            orient=tk.HORIZONTAL,
            command=self.on_detail_changed
        )
        detail_scale.pack(fill=tk.X, pady=2)
        
        self.detail_value_label = ttk.Label(control_frame, text="0.5")
        self.detail_value_label.pack(anchor=tk.E)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Action buttons
        ttk.Button(
            control_frame, 
            text="Process Image", 
            command=self.process_image
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            control_frame, 
            text="Save Processed Image", 
            command=self.save_processed_image
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            control_frame, 
            text="Process Video", 
            command=self.process_full_video
        ).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # File info
        self.file_info_label = ttk.Label(control_frame, text="No file selected", wraplength=250)
        self.file_info_label.pack(anchor=tk.W, pady=5)
        
        self.supported_formats_label = ttk.Label(control_frame, text="", wraplength=250)
        self.supported_formats_label.pack(anchor=tk.W, pady=5)
        
        # Right panel - Preview
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for original and processed views
        self.preview_notebook = ttk.Notebook(preview_frame)
        self.preview_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original image tab
        self.original_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.original_frame, text="Original")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg="black")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Processed image tab
        self.processed_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.processed_frame, text="Processed")
        
        self.processed_canvas = tk.Canvas(self.processed_frame, bg="black")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind resize event to update canvas
        self.root.bind("<Configure>", self.on_resize)
    
    def open_image_file(self):
        """Open an image file for processing"""
        formats = self.file_processor.get_supported_formats()
        filetypes = [
            ("Image files", " ".join(f"*{fmt}" for fmt in formats['image'])),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=filetypes
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.current_file_type = 'image'
        self.video_info = None
        
        # Load the image
        self.original_image = self.file_processor.load_image(file_path)
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not load image: {file_path}")
            return
        
        # Update file info
        file_name = os.path.basename(file_path)
        h, w = self.original_image.shape[:2]
        self.file_info_label.config(
            text=f"Image: {file_name}\nSize: {w}x{h} pixels"
        )
        
        # Display original image
        self.display_image(self.original_image, self.original_canvas)
        
        # Process and display with current effect
        self.process_image()
        
        self.status_var.set(f"Loaded image: {file_name}")
    
    def open_video_file(self):
        """Open a video file for processing"""
        formats = self.file_processor.get_supported_formats()
        filetypes = [
            ("Video files", " ".join(f"*{fmt}" for fmt in formats['video'])),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.current_file_type = 'video'
        
        # Get video info
        self.video_info = self.file_processor.get_video_info(file_path)
        if self.video_info is None:
            messagebox.showerror("Error", f"Could not load video: {file_path}")
            return
        
        # Update file info
        file_name = os.path.basename(file_path)
        w = self.video_info['width']
        h = self.video_info['height']
        duration = self.video_info['duration']
        fps = self.video_info['fps']
        frames = self.video_info['frame_count']
        
        self.file_info_label.config(
            text=f"Video: {file_name}\n"
                f"Size: {w}x{h} pixels\n"
                f"Duration: {duration:.2f} seconds\n"
                f"FPS: {fps:.2f}\n"
                f"Frames: {frames}"
        )
        
        # Extract first frame for preview
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "Could not read video frame")
            return
        
        self.original_image = frame
        
        # Display original frame
        self.display_image(self.original_image, self.original_canvas)
        
        # Process and display with current effect
        self.process_image()
        
        self.status_var.set(f"Loaded video: {file_name}")
    
    def process_image(self):
        """Process the current image/video frame with selected effect"""
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please open an image or video first")
            return
        
        effect = self.effect_var.get()
        self.current_effect = effect
        
        # Set ASCII detail level
        detail = self.detail_var.get()
        self.effect_processor.set_ascii_detail(detail)
        
        # Process the image
        self.status_var.set(f"Processing with {effect}...")
        self.root.update()
        
        start_time = time.time()
        self.processed_image = self.file_processor.process_image(
            self.original_image, effect
        )
        processing_time = time.time() - start_time
        
        # Display processed image
        self.display_image(self.processed_image, self.processed_canvas)
        self.preview_notebook.select(1)  # Switch to processed tab
        
        self.status_var.set(
            f"Applied {effect} effect (took {processing_time:.2f} seconds)"
        )
    
    def process_full_video(self):
        """Process the entire video with selected effect"""
        if not self.current_file or self.current_file_type != 'video':
            messagebox.showinfo("No Video", "Please open a video file first")
            return
        
        if self.processing_video:
            messagebox.showinfo("Processing", "Video processing already in progress")
            return
        
        effect = self.effect_var.get()
        
        # Ask for output file
        file_name = os.path.basename(self.current_file)
        base_name = os.path.splitext(file_name)[0]
        suggested_name = f"{base_name}_{effect}.mp4"
        
        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            initialfile=suggested_name,
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Set ASCII detail level
        detail = self.detail_var.get()
        self.effect_processor.set_ascii_detail(detail)
        
        # Show processing dialog
        processing_window = tk.Toplevel(self.root)
        processing_window.title("Processing Video")
        processing_window.geometry("400x150")
        processing_window.transient(self.root)
        processing_window.grab_set()
        
        ttk.Label(
            processing_window,
            text=f"Processing video with {effect} effect...\nThis may take a while.",
            justify=tk.CENTER
        ).pack(pady=20)
        
        progress = ttk.Progressbar(processing_window, mode="indeterminate")
        progress.pack(fill=tk.X, padx=20, pady=10)
        progress.start()
        
        # Function to run processing in a separate thread
        def process_video_thread():
            self.processing_video = True
            success = self.file_processor.process_video(
                self.current_file, 
                effect_name=effect,
                output_path=output_path
            )
            
            self.processing_video = False
            processing_window.destroy()
            
            if success:
                messagebox.showinfo(
                    "Success", 
                    f"Video processed successfully!\nSaved to: {output_path}"
                )
                self.status_var.set(f"Video processed successfully: {os.path.basename(output_path)}")
            else:
                messagebox.showerror(
                    "Error", 
                    "Failed to process video. Check console for details."
                )
                self.status_var.set("Error processing video")
        
        # Start processing in a separate thread
        import threading
        thread = threading.Thread(target=process_video_thread)
        thread.daemon = True
        thread.start()
    
    def save_processed_image(self):
        """Save the processed image to a file"""
        if self.processed_image is None:
            messagebox.showinfo("No Image", "No processed image to save")
            return
        
        # Get default filename
        if self.current_file:
            file_name = os.path.basename(self.current_file)
            base_name = os.path.splitext(file_name)[0]
            suggested_name = f"{base_name}_{self.current_effect}.png"
        else:
            suggested_name = f"pyart_{self.current_effect}_{get_timestamp()}.png"
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            initialfile=suggested_name,
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        # Save the image
        try:
            cv2.imwrite(save_path, self.processed_image)
            messagebox.showinfo("Success", f"Image saved to: {save_path}")
            self.status_var.set(f"Image saved: {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")
            self.status_var.set("Error saving image")
    
    def display_image(self, image, canvas):
        """Display an image on a canvas with proper scaling"""
        if image is None:
            return
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas is not yet realized, use default size
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 640
            canvas_height = 480
        
        # Calculate aspect ratios
        img_height, img_width = image.shape[:2]
        img_aspect = img_width / img_height
        canvas_aspect = canvas_width / canvas_height
        
        # Calculate dimensions to maintain aspect ratio
        if img_aspect > canvas_aspect:
            # Image is wider than canvas
            new_width = canvas_width
            new_height = int(canvas_width / img_aspect)
        else:
            # Image is taller than canvas
            new_height = canvas_height
            new_width = int(canvas_height * img_aspect)
        
        # Resize image for display
        resized = cv2.resize(
            image, 
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Convert to PIL format
        if len(resized.shape) == 3:
            # Color image (BGR to RGB)
            display_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale image
            display_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(display_image)
        photo_img = ImageTk.PhotoImage(pil_img)
        
        # Clear canvas and display image
        canvas.delete("all")
        
        # Calculate centering position
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Keep a reference to prevent garbage collection
        canvas.image = photo_img
        canvas.create_image(x, y, anchor=tk.NW, image=photo_img)
    
    def on_effect_changed(self, event):
        """Handle effect selection change"""
        self.current_effect = self.effect_var.get()
        
        if self.original_image is not None:
            self.process_image()
    
    def on_detail_changed(self, value):
        """Handle detail level change"""
        detail = self.detail_var.get()
        self.detail_value_label.config(text=f"{detail:.1f}")
        
        # Throttled processing to avoid lag during slider movement
        self.root.after_cancel(self.on_detail_changed_throttled) if hasattr(self, 'on_detail_changed_throttled') else None
        self.on_detail_changed_throttled = self.root.after(200, self.process_image)
    
    def on_resize(self, event):
        """Handle window resize events"""
        # Only update if we already have images loaded
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
        
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_canvas)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        PyArt - File Converter

        Convert images and videos to ASCII art
        
        Part of the PyArt project
        
        Use the various ASCII art effects to transform
        your images and videos into ASCII masterpieces!
        """
        
        messagebox.showinfo("About PyArt File Converter", about_text)
    
    def show_instructions(self):
        """Show instructions dialog"""
        instructions = """
        Instructions:
        
        1. Open an image or video file using the buttons or File menu
        
        2. Select an ASCII effect from the dropdown menu
        
        3. Adjust the ASCII detail level using the slider
        
        4. Click "Process Image" to apply the effect
        
        5. Save the processed image using the "Save" button
        
        6. For videos, click "Process Video" to create a new
           video with the effect applied to each frame
        
        Switch between original and processed views using the tabs.
        """
        
        messagebox.showinfo("Instructions", instructions)


def main():
    """Main entry point for the file converter application"""
    root = tk.Tk()
    app = FileConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
