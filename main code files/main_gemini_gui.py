import cv2
from ultralytics import YOLO
import os
import glob
from PIL import Image, ImageTk
import base64
import torch
import numpy as np
import requests
import json
import time
import tkinter as tk
from tkinter import ttk
import threading

# ============================================================
# CONFIGURATION: Set your Gemini API key here
# ============================================================
GEMINI_API_KEY = "AIzaSyCdqwK47KRGvnPDpxNzH6EkicBFGaAeFKE"  # Replace with your actual API key

# Define inspection prompts for BASE YOLOv8x-World model (uses synonyms)
INSPECTION_ELEMENTS_BASE = {
    "sockets": ["wall socket", "power outlet", "electrical socket", "power point", "wall outlet"],
    "fire_safety": ["fire extinguisher", "fire safety equipment", "fire suppression device"],
    "lighting": ["light fixture", "ceiling light", "wall light", "lamp", "light fitting", "fluorescent tube"]
}

# Define inspection prompts for CUSTOM YOLOv8x model (uses trained class names)
INSPECTION_ELEMENTS_CUSTOM = {
    "sockets": ["outlet"],
    "fire_safety": ["fire extinguisher"],
    "lighting": ["fluorescent tube"]
}

class RobotInspectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Robot Inspector - Automated Facility Inspection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize control variables first (before GUI elements that use them)
        self.paused = False
        self.continue_event = threading.Event()
        self.image_paths = []
        self.selected_image_index = None
        self.thumbnail_labels = []
        self.selected_model = tk.StringVar(value="yolov8x-world.pt")  # Default model
        
        # Create main container
        self.main_frame = tk.Frame(root, bg='#2c3e50')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="AUTOMATED ROBOT INSPECTION SYSTEM",
            font=('Helvetica', 20, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=10)
        
        # Create split view
        content_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Image displays with scrollbar
        left_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create a canvas with scrollbar for scrollable content
        canvas = tk.Canvas(left_frame, bg='#34495e', highlightthickness=0)
        scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#34495e')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Thumbnail gallery in scrollable frame
        gallery_title = tk.Label(
            scrollable_frame,
            text="Image Gallery - Click to Select",
            font=('Helvetica', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        gallery_title.pack(pady=5)
        
        # Create scrollable canvas for thumbnails
        thumb_canvas = tk.Canvas(scrollable_frame, bg='#34495e', height=120, highlightthickness=0)
        thumb_scrollbar = tk.Scrollbar(scrollable_frame, orient="horizontal", command=thumb_canvas.xview)
        self.thumb_frame = tk.Frame(thumb_canvas, bg='#34495e')
        
        self.thumb_frame.bind(
            "<Configure>",
            lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all"))
        )
        
        thumb_canvas.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        thumb_canvas.configure(xscrollcommand=thumb_scrollbar.set)
        
        thumb_canvas.pack(side="top", fill="x", expand=True, padx=10)
        thumb_scrollbar.pack(side="top", fill="x", padx=10)
        
        # Main image view
        image_title = tk.Label(
            scrollable_frame,
            text="Full Image with Detections",
            font=('Helvetica', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        image_title.pack(pady=5)
        
        self.image_label = tk.Label(scrollable_frame, bg='#2c3e50')
        self.image_label.pack(padx=10, pady=5)
        
        # Cropped image view (what Gemini sees)
        cropped_title = tk.Label(
            scrollable_frame,
            text="Cropped View (Sent to Gemini)",
            font=('Helvetica', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        cropped_title.pack(pady=5)
        
        self.cropped_label = tk.Label(scrollable_frame, bg='#2c3e50')
        self.cropped_label.pack(padx=10, pady=5)
        
        # Right side - Robot and status
        right_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RIDGE, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Robot visualization
        robot_title = tk.Label(
            right_frame,
            text="Robot Status",
            font=('Helvetica', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        robot_title.pack(pady=5)
        
        self.robot_canvas = tk.Canvas(right_frame, width=400, height=150, bg='#2c3e50', highlightthickness=0)
        self.robot_canvas.pack(pady=10)
        
        # Draw robot
        self.draw_robot()
        
        # Model selection frame
        model_frame = tk.Frame(right_frame, bg='#34495e')
        model_frame.pack(pady=10)
        
        model_title = tk.Label(
            model_frame,
            text="Detection Model",
            font=('Helvetica', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        model_title.pack()
        
        # Radio buttons for model selection
        radio_frame = tk.Frame(model_frame, bg='#34495e')
        radio_frame.pack(pady=5)
        
        world_radio = tk.Radiobutton(
            radio_frame,
            text="YOLOv11x-Custom",
            variable=self.selected_model,
            value="yolov11x_custom_trained.pt",
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Helvetica', 9)
        )
        world_radio.pack(side=tk.LEFT, padx=5)
        
        yolo11_radio = tk.Radiobutton(
            radio_frame,
            text="YOLOv11x",
            variable=self.selected_model,
            value="other/yolo11x.pt",
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Helvetica', 9)
        )
        yolo11_radio.pack(side=tk.LEFT, padx=5)
        
        custom_radio = tk.Radiobutton(
            radio_frame,
            text="Custom YOLOv8x",
            variable=self.selected_model,
            value="custom_yolov8x.pt",
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Helvetica', 9)
        )
        custom_radio.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            right_frame,
            text="Status: Initializing...",
            font=('Helvetica', 11, 'bold'),
            bg='#34495e',
            fg='#3498db',
            wraplength=400
        )
        self.status_label.pack(pady=10)
        
        # Speech bubble for inspection results
        bubble_frame = tk.Frame(right_frame, bg='#34495e')
        bubble_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        bubble_title = tk.Label(
            bubble_frame,
            text="Inspection Report",
            font=('Helvetica', 11, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        bubble_title.pack()
        
        # Text widget for inspection results with scrollbar
        text_frame = tk.Frame(bubble_frame, bg='#ecf0f1', relief=tk.RAISED, bd=3)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            text_frame,
            font=('Courier', 10),
            bg='#ecf0f1',
            fg='#2c3e50',
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.result_text.yview)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.main_frame,
            orient=tk.HORIZONTAL,
            length=1180,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        
        # Button frame for controls
        button_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        # Start button
        self.start_button = tk.Button(
            button_frame,
            text="START INSPECTION",
            font=('Helvetica', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            command=self.start_inspection,
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Load images on startup
        self.load_image_gallery()
        
    def draw_robot(self):
        """Draw a simple robot visualization"""
        # Robot body
        self.robot_canvas.create_rectangle(150, 50, 250, 120, fill='#3498db', outline='#2980b9', width=3)
        # Robot head
        self.robot_canvas.create_oval(170, 20, 230, 60, fill='#3498db', outline='#2980b9', width=3)
        # Eyes
        self.robot_canvas.create_oval(185, 30, 195, 40, fill='#ecf0f1', outline='#2c3e50', width=2)
        self.robot_canvas.create_oval(205, 30, 215, 40, fill='#ecf0f1', outline='#2c3e50', width=2)
        # Pupils (can be animated)
        self.left_pupil = self.robot_canvas.create_oval(188, 33, 192, 37, fill='#2c3e50')
        self.right_pupil = self.robot_canvas.create_oval(208, 33, 212, 37, fill='#2c3e50')
        # Antenna
        self.robot_canvas.create_line(200, 20, 200, 5, fill='#2980b9', width=3)
        self.robot_canvas.create_oval(195, 0, 205, 10, fill='#e74c3c', outline='#c0392b', width=2)
        # Arms
        self.robot_canvas.create_rectangle(130, 60, 150, 90, fill='#3498db', outline='#2980b9', width=2)
        self.robot_canvas.create_rectangle(250, 60, 270, 90, fill='#3498db', outline='#2980b9', width=2)
        # Wheels
        self.robot_canvas.create_oval(160, 115, 190, 140, fill='#34495e', outline='#2c3e50', width=2)
        self.robot_canvas.create_oval(210, 115, 240, 140, fill='#34495e', outline='#2c3e50', width=2)
        
    def animate_robot_thinking(self):
        """Animate robot eyes to show it's thinking"""
        for _ in range(3):
            # Move pupils left
            self.robot_canvas.coords(self.left_pupil, 186, 33, 190, 37)
            self.robot_canvas.coords(self.right_pupil, 206, 33, 210, 37)
            self.root.update()
            time.sleep(0.2)
            # Move pupils right
            self.robot_canvas.coords(self.left_pupil, 190, 33, 194, 37)
            self.robot_canvas.coords(self.right_pupil, 210, 33, 214, 37)
            self.root.update()
            time.sleep(0.2)
        # Center pupils
        self.robot_canvas.coords(self.left_pupil, 188, 33, 192, 37)
        self.robot_canvas.coords(self.right_pupil, 208, 33, 212, 37)
        
    def update_status(self, message, color='#3498db'):
        """Update the status label"""
        self.status_label.config(text=f"Status: {message}", fg=color)
        self.root.update()
        
    def add_result(self, message, tag='normal'):
        """Add a message to the result text widget"""
        self.result_text.insert(tk.END, message + "\n\n")
        if tag == 'ok':
            # Find and tag the last insertion
            start = self.result_text.search("[OK]", tk.END, backwards=True)
            if start:
                self.result_text.tag_add("ok", start, f"{start} lineend")
                self.result_text.tag_config("ok", foreground='#27ae60', font=('Courier', 10, 'bold'))
        elif tag == 'faulty':
            start = self.result_text.search("[FAULTY]", tk.END, backwards=True)
            if start:
                self.result_text.tag_add("faulty", start, f"{start} lineend")
                self.result_text.tag_config("faulty", foreground='#e74c3c', font=('Courier', 10, 'bold'))
        
        self.result_text.see(tk.END)
        self.root.update()
        
    def display_image(self, image_path):
        """Display an image in the GUI"""
        try:
            img = Image.open(image_path)
            # Resize to fit
            img.thumbnail((550, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            self.root.update()
        except Exception as e:
            print(f"Error displaying image: {e}")
            
    def display_cropped_image(self, image_path):
        """Display the cropped image that will be sent to Gemini"""
        try:
            img = Image.open(image_path)
            # Stretch to match full image size (no aspect ratio preservation)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.cropped_label.config(image=photo)
            self.cropped_label.image = photo  # Keep a reference
            self.root.update()
        except Exception as e:
            print(f"Error displaying cropped image: {e}")
            
    def clear_cropped_image(self):
        """Clear the cropped image display"""
        self.cropped_label.config(image='')
        self.cropped_label.image = None
        self.root.update()
            
    def start_inspection(self):
        """Start the inspection process in a separate thread"""
        if self.selected_image_index is None:
            self.update_status("Please select an image first!", '#e74c3c')
            return
        
        # Clear previous cropped image when starting new inspection
        self.clear_cropped_image()
            
        self.start_button.config(state=tk.DISABLED, bg='#7f8c8d')
        self.result_text.delete(1.0, tk.END)
        self.add_result(">>> INSPECTION STARTED")
        self.add_result("="*50)
        
        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.run_inspection, daemon=True)
        thread.start()
        
    def run_inspection(self):
        """Main inspection logic"""
        try:
            # Check if images are loaded
            if not self.image_paths:
                self.update_status(f"Error: No images found!", '#e74c3c')
                self.add_result(f"[ERROR] No images found in 'gemini_test_folder'")
                return
            
            # Use selected image
            image_path = self.image_paths[self.selected_image_index]
            
            self.add_result(f"[INFO] Processing selected image")
            
            # Setup GPU
            device = 0 if torch.cuda.is_available() else 'cpu'
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Get detection prompts based on selected model
            model_path = self.selected_model.get()
            
            # Choose the appropriate prompt set
            if "custom" in model_path.lower():
                INSPECTION_ELEMENTS = INSPECTION_ELEMENTS_CUSTOM
                self.add_result(f"[INFO] Using CUSTOM model prompts: {', '.join(sum(INSPECTION_ELEMENTS.values(), []))}")
            else:
                INSPECTION_ELEMENTS = INSPECTION_ELEMENTS_BASE
                self.add_result(f"[INFO] Using BASE model prompts (open-vocabulary): {', '.join(sum(INSPECTION_ELEMENTS.values(), []))}")
            
            all_prompts = []
            for prompts in INSPECTION_ELEMENTS.values():
                all_prompts.extend(prompts)
            
            self.add_result("="*50)
            self.add_result(f"[IMAGE] {os.path.basename(image_path)}")
            self.update_status(f"Processing selected image...", '#3498db')
            
            # Display the annotated image if it exists, otherwise the original
            annotated_path = os.path.join('gemini_results', f"annotated_{os.path.basename(image_path)}")
            
            # Run detection
            self.update_status(f"Detecting objects...", '#f39c12')
            model = YOLO(model_path)
            
            # Only use set_classes() for YOLOv8-World models (open-vocabulary detection)
            # Standard YOLO models (like YOLOv11) and custom models don't support set_classes()
            if "world" in model_path.lower():
                model.set_classes(all_prompts)
                self.add_result(f"[INFO] YOLOv8x-World: Using open-vocabulary detection with {len(all_prompts)} synonym prompts")
            elif "custom" in model_path.lower():
                self.add_result(f"[INFO] Custom Model: Using native trained classes")
            else:
                self.add_result(f"[INFO] Standard YOLO Model: Using pre-trained COCO classes")
            
            detections = self.detect_and_save(model, image_path, device=device)
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Display the annotated image
            if os.path.exists(annotated_path):
                self.display_image(annotated_path)
            else:
                self.display_image(image_path)
            
            if not detections:
                self.add_result("[WARNING] No elements detected in this image.")
                self.progress['value'] = 1
                self.update_status("No elements detected!", '#e67e22')
                return
            
            self.add_result(f"[DETECTED] Found {len(detections)} detection(s)")
            
            # Categorize detections
            categorized_detections = {}
            for detection in detections:
                element_name = detection["label"]
                category = self.get_element_category(element_name)
                if not category:
                    category = element_name
                
                if category not in categorized_detections:
                    categorized_detections[category] = {
                        "category": category,
                        "image_path": image_path,
                        "detected_label": element_name,
                        "confidence": detection["confidence"],
                        "box": detection["box"]
                    }
            
            # Inspect each detection
            for idx, info in enumerate(categorized_detections.values(), 1):
                category = info["category"]
                self.update_status(f"Inspecting {category}... ({idx}/{len(categorized_detections)})", '#9b59b6')
                self.add_result(f"\n[ROBOT] Moving to: {info['detected_label']} (confidence: {info['confidence']:.2f})")
                
                # Animate robot
                self.animate_robot_thinking()
                
                # Inspect with Gemini (this will also display the cropped image)
                result = self.inspect_element_gemini(
                    category,
                    info["image_path"],
                    crop_box=info["box"]
                )
                
                # Display result
                r = result.lower().strip()
                if r.startswith("ok") or "good condition" in r:
                    self.add_result(f"[OK] {category}: {result}", 'ok')
                elif r.startswith("faulty") or "defect" in r or "damage" in r or "fault" in r:
                    self.add_result(f"[FAULTY] {category}: {result}", 'faulty')
                else:
                    self.add_result(f"[UNKNOWN] {category}: {result}")
                
                self.progress['value'] = idx
            
            # Completed
            self.add_result("="*50)
            self.add_result(f"[COMPLETE] Inspection complete!")
            self.update_status("Inspection Complete!", '#27ae60')
            
        except Exception as e:
            self.add_result(f"[ERROR] {str(e)}")
            self.update_status(f"Error occurred!", '#e74c3c')
            import traceback
            traceback.print_exc()
        finally:
            self.start_button.config(state=tk.NORMAL, bg='#27ae60')
            
    def get_element_category(self, detected_label):
        """Map a detected label back to its category"""
        detected_label = detected_label.lower()
        
        # Check both prompt sets (base and custom) to find the category
        # This allows the method to work regardless of which model was used
        for category, prompts in INSPECTION_ELEMENTS_BASE.items():
            if any(prompt.lower() in detected_label for prompt in prompts):
                return category
        
        for category, prompts in INSPECTION_ELEMENTS_CUSTOM.items():
            if any(prompt.lower() in detected_label for prompt in prompts):
                return category
                
        return None
        
    def crop_object(self, image_path, box, padding=10):
        """Crop the detected object from the image with optional padding"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        cropped = img[y1:y2, x1:x2]
        return cropped
        
    def inspect_element_gemini(self, element_name, image_path, crop_box=None):
        """Use Gemini 2.5 Flash to inspect the condition of a detected element"""
        api_key = GEMINI_API_KEY
        
        crop_path = None
        if crop_box is not None:
            cropped_img = self.crop_object(image_path, crop_box)
            if cropped_img is not None:
                crop_path = os.path.join('gemini_results', f'cropped_{element_name}_{os.path.basename(image_path)}')
                os.makedirs('gemini_results', exist_ok=True)
                cv2.imwrite(crop_path, cropped_img)
                image_to_inspect = crop_path
                
                # Display the cropped image that will be sent to Gemini
                self.display_cropped_image(crop_path)
            else:
                image_to_inspect = image_path
        else:
            image_to_inspect = image_path

        with open(image_to_inspect, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = f"""You are an inspection robot. You must ONLY respond in one of these two formats:
1. 'OK: [element] in good condition' if no issues are found
2. 'FAULTY: [brief reason]' if you spot any defects

No other response format is allowed. No explanations or commentary.

Inspect this {element_name}. Is it in good condition or faulty? Remember to use ONLY the required response format."""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }]
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
            else:
                return "ERROR: No response from Gemini API"
                
        except Exception as e:
            return f"ERROR: {str(e)}"
            
    def detect_and_save(self, model, image_path, save_dir='gemini_results', device=0):
        """Run detection and save annotated image"""
        os.makedirs(save_dir, exist_ok=True)
        
        results = model.predict(source=image_path, conf=0.30, verbose=False, device=device)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            img = cv2.imread(image_path)
            
            # First, collect all detections with their information
            all_detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                all_detections.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": float(confidence),
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
            
            # Filter overlapping boxes of the same class (Non-Maximum Suppression)
            filtered_detections = self.filter_overlapping_detections(all_detections)
            
            # Draw filtered detections
            for detection in filtered_detections:
                x1, y1, x2, y2 = detection["box"]
                confidence = detection["confidence"]
                class_name = detection["class_name"]
                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(img, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                detections.append({
                    "label": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })
            
            output_path = os.path.join(save_dir, f"annotated_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, img)
        
        return detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def filter_overlapping_detections(self, detections, iou_threshold=0.3):
        """
        Filter out overlapping detections of the same class, keeping only the highest confidence one.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold to consider boxes as overlapping (default: 0.3)
        
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        # Group detections by class name
        class_groups = {}
        for detection in sorted_detections:
            class_name = detection["class_name"]
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        # Apply NMS for each class separately
        filtered = []
        for class_name, class_detections in class_groups.items():
            keep = []
            for i, det1 in enumerate(class_detections):
                should_keep = True
                for det2 in keep:
                    iou = self.calculate_iou(det1["box"], det2["box"])
                    if iou > iou_threshold:
                        # Overlaps with a higher confidence detection
                        should_keep = False
                        break
                
                if should_keep:
                    keep.append(det1)
            
            filtered.extend(keep)
        
        return filtered
    
    def load_image_gallery(self):
        """Load all images and display thumbnails"""
        test_images_dir = "gemini_test_folder"
        if not os.path.exists(test_images_dir):
            self.update_status(f"Warning: '{test_images_dir}' not found!", '#e67e22')
            return
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
        
        if not self.image_paths:
            self.update_status(f"Warning: No images found in '{test_images_dir}'", '#e67e22')
            return
        
        # Create thumbnails
        for idx, img_path in enumerate(self.image_paths):
            try:
                img = Image.open(img_path)
                img.thumbnail((100, 100), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Create a frame for each thumbnail
                thumb_container = tk.Frame(self.thumb_frame, bg='#2c3e50', relief=tk.RAISED, bd=2)
                thumb_container.pack(side=tk.LEFT, padx=5, pady=5)
                
                # Image label
                img_label = tk.Label(thumb_container, image=photo, bg='#2c3e50', cursor='hand2')
                img_label.image = photo  # Keep reference
                img_label.pack()
                
                # Filename label
                name_label = tk.Label(
                    thumb_container, 
                    text=os.path.basename(img_path)[:15] + '...' if len(os.path.basename(img_path)) > 15 else os.path.basename(img_path),
                    bg='#2c3e50',
                    fg='#ecf0f1',
                    font=('Helvetica', 8)
                )
                name_label.pack()
                
                # Bind click event
                img_label.bind('<Button-1>', lambda e, i=idx: self.select_image(i))
                name_label.bind('<Button-1>', lambda e, i=idx: self.select_image(i))
                
                self.thumbnail_labels.append(thumb_container)
                
            except Exception as e:
                print(f"Error loading thumbnail for {img_path}: {e}")
        
        self.update_status(f"Loaded {len(self.image_paths)} images. Click on a thumbnail to select.", '#3498db')
        self.progress['maximum'] = len(self.image_paths)
    
    def select_image(self, index):
        """Select an image from the gallery"""
        # Reset all thumbnail borders
        for thumb in self.thumbnail_labels:
            thumb.config(relief=tk.RAISED, bd=2, bg='#2c3e50')
        
        # Highlight selected thumbnail
        self.thumbnail_labels[index].config(relief=tk.SUNKEN, bd=4, bg='#3498db')
        
        self.selected_image_index = index
        self.update_status(f"Selected: {os.path.basename(self.image_paths[index])}", '#27ae60')
        
        # Enable start button
        self.start_button.config(state=tk.NORMAL, bg='#27ae60')

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotInspectorGUI(root)
    root.mainloop()
