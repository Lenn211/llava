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
GEMINI_API_KEY = "AIzaSyCoqFjIYcg2cr-nBxTGqG4ARTyoptpnx1I"  # Replace with your actual API key

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
        self.root.title("Robot Inspector - Video Input + Automated Inspection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize control variables
        self.paused = False
        self.continue_event = threading.Event()
        self.image_paths = []
        self.selected_image_index = None
        self.thumbnail_labels = []
        self.selected_model = tk.StringVar(value="yolov11x_custom_trained_v2.pt")
        self.dual_model_mode = tk.BooleanVar(value=False)  # Toggle for switching between models
        
        # Camera variables
        self.camera_active = False
        self.camera = None
        self.camera_thread = None
        self.current_frame = None
        self.current_detections = []
        self.user_is_seeking = False
        self.seek_frame = None
        
        # TRACKING: Store inspected objects
        # Format: {'class': 'name', 'box': [x1,y1,x2,y2], 'status': 'OK'/'FAULTY', 'last_seen': timestamp}
        self.inspected_objects = []
        
        self.detection_model = None
        self.camera_source = tk.StringVar(value="IMG_2196(1).MOV")  # Default to video file
        
        # Create main container
        self.main_frame = tk.Frame(root, bg='#2c3e50')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Robot Inspector - Video Input",
            font=('Arial', 14, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=5)
        
        # Create split view
        content_frame = tk.Frame(self.main_frame, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Camera/Image displays
        left_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Camera controls
        camera_control_frame = tk.Frame(left_frame, bg='#34495e')
        camera_control_frame.pack(pady=5)
        
        tk.Label(
            camera_control_frame,
            text="Video Source:",
            font=('Arial', 9),
            bg='#34495e',
            fg='#ecf0f1'
        ).pack(side=tk.LEFT, padx=5)
        
        # Camera source entry
        camera_entry = tk.Entry(
            camera_control_frame,
            textvariable=self.camera_source,
            width=25,
            font=('Arial', 9)
        )
        camera_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            camera_control_frame,
            text="(Video file path)",
            font=('Arial', 8),
            bg='#34495e',
            fg='#95a5a6'
        ).pack(side=tk.LEFT, padx=5)
        
        # Test connection button
        test_button = tk.Button(
            camera_control_frame,
            text="TEST",
            font=('Arial', 8),
            bg='#3498db',
            fg='white',
            command=self.test_camera_connection,
            relief=tk.RAISED,
            bd=2,
            padx=8,
            pady=3
        )
        test_button.pack(side=tk.LEFT, padx=3)
        
        # Camera start/stop button
        self.camera_button = tk.Button(
            camera_control_frame,
            text="START",
            font=('Arial', 9, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.toggle_camera,
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=3
        )
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        # Live camera view
        camera_title = tk.Label(
            left_frame,
            text="Video Feed",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        camera_title.pack(pady=2)
        
        # Use a frame to contain the camera label for better sizing
        camera_frame = tk.Frame(left_frame, bg='#2c3e50')
        camera_frame.pack(padx=5, pady=2, fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(camera_frame, bg='#2c3e50', text="Video stopped")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Video Progress Slider
        self.progress_slider = tk.Scale(
            left_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            bg='#34495e', 
            fg='#ecf0f1',
            highlightthickness=0,
            showvalue=0
        )
        self.progress_slider.pack(fill=tk.X, padx=5, pady=0)
        
        # Bind slider events for seeking
        self.progress_slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        # Cropped image view (inspection target)
        cropped_title = tk.Label(
            left_frame,
            text="Inspection Target",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        cropped_title.pack(pady=2)
        
        # Frame for cropped image with fixed size
        cropped_frame = tk.Frame(left_frame, bg='#2c3e50', width=600, height=450)
        cropped_frame.pack(padx=5, pady=2)
        cropped_frame.pack_propagate(False)  # Prevent resizing
        
        self.cropped_label = tk.Label(cropped_frame, bg='#2c3e50', text="No selection")
        self.cropped_label.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Robot, controls, and status
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
        model_frame.pack(pady=5)
        
        model_title = tk.Label(
            model_frame,
            text="Model",
            font=('Arial', 9, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        model_title.pack()
        
        # Radio buttons for model selection
        radio_frame = tk.Frame(model_frame, bg='#34495e')
        radio_frame.pack(pady=3)
        
        world_radio = tk.Radiobutton(
            radio_frame,
            text="YOLOv8x-World",
            variable=self.selected_model,
            value="yolov8x-world.pt",
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Arial', 8)
        )
        world_radio.pack(side=tk.LEFT, padx=3)
        
        custom_radio = tk.Radiobutton(
            radio_frame,
            text="Custom YOLOv11x",
            variable=self.selected_model,
            value="yolov11x_custom_trained_v2.pt",
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Arial', 8)
        )
        custom_radio.pack(side=tk.LEFT, padx=3)

        custom_world_radio = tk.Radiobutton(
            radio_frame,
            text="Custom YOLOv8x-World",
            variable=self.selected_model,
            value="custom_yolov8x-world.pt",
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Arial', 8)
        )
        custom_world_radio.pack(side=tk.LEFT, padx=3)
        
        # Dual Model Mode Checkbox
        dual_check = tk.Checkbutton(
            model_frame,
            text="Dual Model Mode (Switching)",
            variable=self.dual_model_mode,
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#3498db',
            font=('Arial', 8)
        )
        dual_check.pack(pady=2)
        
        # Detection list (clickable)
        detection_title = tk.Label(
            right_frame,
            text="Detections",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        detection_title.pack(pady=3)
        
        # Listbox for detections
        list_frame = tk.Frame(right_frame, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        list_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=3)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detection_listbox = tk.Listbox(
            list_frame,
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#2c3e50',
            yscrollcommand=scrollbar.set,
            height=5,
            relief=tk.FLAT,
            selectmode=tk.SINGLE
        )
        self.detection_listbox.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        scrollbar.config(command=self.detection_listbox.yview)
        
        # Bind selection event
        self.detection_listbox.bind('<<ListboxSelect>>', self.on_detection_select)
        
        # Inspect button (triggers Gemini inspection)
        self.inspect_button = tk.Button(
            right_frame,
            text="INSPECT",
            font=('Arial', 10, 'bold'),
            bg='#e67e22',
            fg='white',
            activebackground='#d35400',
            command=self.inspect_selected_detection,
            relief=tk.RAISED,
            bd=2,
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        self.inspect_button.pack(pady=5)
        
        # TEST BUTTON - to verify add_result works
        test_button = tk.Button(
            right_frame,
            text="TEST REPORT",
            font=('Arial', 8),
            bg='#9b59b6',
            fg='white',
            command=lambda: self.add_result("TEST: This is a test message!"),
            relief=tk.RAISED,
            bd=1,
            padx=8,
            pady=3
        )
        test_button.pack(pady=2)
        
        # Status label
        self.status_label = tk.Label(
            right_frame,
            text="Ready",
            font=('Arial', 10),
            bg='#34495e',
            fg='#3498db',
            wraplength=350
        )
        self.status_label.pack(pady=5)
        
        # Inspection results
        bubble_title = tk.Label(
            right_frame,
            text="Report",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        bubble_title.pack()
        
        # Text widget for inspection results with scrollbar
        text_frame = tk.Frame(right_frame, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=3)
        
        scrollbar2 = tk.Scrollbar(text_frame)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            text_frame,
            font=('Arial', 8),
            bg='#ecf0f1',
            fg='#2c3e50',
            wrap=tk.WORD,
            yscrollcommand=scrollbar2.set,
            relief=tk.FLAT,
            height=8
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        scrollbar2.config(command=self.result_text.yview)
        
        # Add a test message to verify widget works
        self.result_text.insert(tk.END, "Report will appear here...\n")
        self.result_text.tag_config("info", foreground='#7f8c8d', font=('Arial', 8, 'italic'))
        print("[INIT] Report text widget created and initialized")
        print(f"[INIT] Initial content: {self.result_text.get('1.0', tk.END)}")
        
    def on_slider_press(self, event):
        """User started dragging the slider"""
        self.user_is_seeking = True

    def on_slider_release(self, event):
        """User released the slider - perform seek"""
        self.user_is_seeking = False
        value = self.progress_slider.get()
        self.seek_frame = int(value)
        print(f"[SEEK] Seeking to frame {self.seek_frame}")

    def draw_robot(self):
        """Draw a simple robot visualization"""
        # Robot body
        self.robot_canvas.create_rectangle(150, 50, 250, 120, fill='#3498db', outline='#2980b9', width=3)
        # Robot head
        self.robot_canvas.create_oval(170, 20, 230, 60, fill='#3498db', outline='#2980b9', width=3)
        # Eyes
        self.robot_canvas.create_oval(185, 30, 195, 40, fill='#ecf0f1', outline='#2c3e50', width=2)
        self.robot_canvas.create_oval(205, 30, 215, 40, fill='#ecf0f1', outline='#2c3e50', width=2)
        # Pupils
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
        """Animate robot eyes to show it's thinking - thread-safe"""
        def _animate_step(step):
            if step >= 6:  # 3 cycles * 2 positions
                # Reset to center
                self.robot_canvas.coords(self.left_pupil, 188, 33, 192, 37)
                self.robot_canvas.coords(self.right_pupil, 208, 33, 212, 37)
                return
            
            if step % 2 == 0:
                # Move left
                self.robot_canvas.coords(self.left_pupil, 186, 33, 190, 37)
                self.robot_canvas.coords(self.right_pupil, 206, 33, 210, 37)
            else:
                # Move right
                self.robot_canvas.coords(self.left_pupil, 190, 33, 194, 37)
                self.robot_canvas.coords(self.right_pupil, 210, 33, 214, 37)
            
            # Schedule next step
            self.root.after(200, lambda: _animate_step(step + 1))
        
        # Start animation from main thread
        self.root.after(0, lambda: _animate_step(0))
        
    def update_status(self, message, color='#3498db'):
        """Update the status label - thread-safe"""
        def _update():
            try:
                self.status_label.config(text=message, fg=color)
                self.status_label.update_idletasks()
                print(f"[DEBUG] Status updated: {message}")
            except Exception as e:
                print(f"[ERROR] Failed to update status: {e}")
        
        # Always use after() to ensure thread safety
        self.root.after(0, _update)
        
    def add_result(self, message, tag='normal'):
        """Add a message to the result text widget - thread-safe"""
        def _add():
            try:
                print(f"[DEBUG] _add() called with message: {message[:50]}...")
                print(f"[DEBUG] result_text widget exists: {self.result_text is not None}")
                print(f"[DEBUG] result_text widget type: {type(self.result_text)}")
                
                # Get current content before insert
                current_content = self.result_text.get("1.0", tk.END)
                print(f"[DEBUG] Current content length: {len(current_content)}")
                
                self.result_text.insert(tk.END, message + "\n")
                
                # Get content after insert
                new_content = self.result_text.get("1.0", tk.END)
                print(f"[DEBUG] New content length: {len(new_content)}")
                
                if tag == 'ok':
                    start = self.result_text.search("[OK]", tk.END, backwards=True)
                    if start:
                        self.result_text.tag_add("ok", start, f"{start} lineend")
                        self.result_text.tag_config("ok", foreground='#27ae60', font=('Arial', 8, 'bold'))
                elif tag == 'faulty':
                    start = self.result_text.search("[FAULTY]", tk.END, backwards=True)
                    if start:
                        self.result_text.tag_add("faulty", start, f"{start} lineend")
                        self.result_text.tag_config("faulty", foreground='#e74c3c', font=('Arial', 8, 'bold'))
                
                self.result_text.see(tk.END)
                self.result_text.update_idletasks()
                print(f"[DEBUG] Successfully added to result_text: {message}")
            except Exception as e:
                print(f"[ERROR] Failed to add result: {e}")
                import traceback
                traceback.print_exc()
        
        # Always use after() to ensure thread safety
        print(f"[DEBUG] add_result() called, scheduling _add() via root.after()")
        self.root.after(0, _add)
        
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Start the camera feed"""
        try:
            # Parse camera source
            source = self.camera_source.get()
            original_source = source
            try:
                # Try as integer (webcam index)
                source = int(source)
                print(f"[INFO] Using webcam index: {source}")
            except ValueError:
                # Use as string (URL or device path)
                print(f"[INFO] Using camera URL/path: {source}")
                pass
            
            # Try to open camera with detailed diagnostics
            print(f"[INFO] Opening camera with source: {source}")
            self.update_status("Opening video...", '#f39c12')
            
            # For network streams, try auto-detection
            if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp')):
                print(f"[INFO] Network stream detected, trying auto-detection...")
                self.update_status("Auto-detecting URL format...", '#f39c12')
                
                success, working_url, test_frame = self.try_multiple_urls(source)
                
                if success:
                    print(f"[INFO] Auto-detected working URL: {working_url}")
                    source = working_url
                    # Update the camera source field
                    if working_url != original_source:
                        self.camera_source.set(working_url)
                    
                    # Re-open with the working URL
                    self.camera = cv2.VideoCapture(source)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    print(f"[INFO] Auto-detection failed, trying original URL...")
                    self.camera = cv2.VideoCapture(source)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.camera = cv2.VideoCapture(source)
            
            # Configure slider for video
            total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                self.progress_slider.config(to=total_frames)
                print(f"[INFO] Video length: {total_frames} frames")
            
            # For network streams, set buffer size and timeout
            if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp')):
                print(f"[INFO] Network stream detected, configuring...")
                # Reduce buffer size for lower latency
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Some backends support timeout settings
                try:
                    # Try setting a connection timeout (not all backends support this)
                    self.camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    self.camera.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                except:
                    pass
            
            if not self.camera.isOpened():
                error_msg = f"Failed to open video: {original_source}"
                print(f"[ERROR] {error_msg}")
                self.update_status(error_msg, '#e74c3c')
                
                # Provide helpful suggestions
                if isinstance(source, str) and source.startswith('http'):
                    print(f"[HINT] For WiFi cameras, try:")
                    print(f"  - Check phone and PC are on same WiFi network")
                    print(f"  - Verify IP address in camera app")
                    print(f"  - Try TEST button first")
                    print(f"  - Run: python test_wifi_camera.py <IP_ADDRESS>")
                    print(f"  - See: WIFI_CAMERA_TROUBLESHOOTING.md")
                
                return
            
            print(f"[INFO] Video opened successfully")
            
            # Test reading a frame to verify connection
            self.update_status("Testing video connection...", '#f39c12')
            ret, test_frame = self.camera.read()
            
            if not ret or test_frame is None:
                error_msg = f"Video opened but cannot read frames: {original_source}"
                print(f"[ERROR] {error_msg}")
                self.update_status(error_msg, '#e74c3c')
                self.camera.release()
                self.camera = None
                
                if isinstance(source, str) and source.startswith('http'):
                    print(f"[HINT] Camera connection issue - try:")
                    print(f"  - Different URL format (e.g., /video, /videofeed, /mjpegfeed)")
                    print(f"  - Check camera app is running and streaming")
                    print(f"  - Use TEST button to find working URL")
                    print(f"  - See: WIFI_CAMERA_TROUBLESHOOTING.md")
                
                return
            
            print(f"[INFO] Successfully read test frame: {test_frame.shape}")
            
            # Load detection model
            self.update_status("Loading detection model...", '#f39c12')
            device = 0 if torch.cuda.is_available() else 'cpu'
            
            if self.dual_model_mode.get():
                print("[INFO] Dual Model Mode: Loading both YOLOv8x-World and YOLOv11x-Custom")
                self.update_status("Loading Dual Models...", '#f39c12')
                
                # Load World Model
                self.model_world = YOLO('yolov8x-world.pt')
                all_prompts = []
                for prompts in INSPECTION_ELEMENTS_BASE.values():
                    all_prompts.extend(prompts)
                self.model_world.set_classes(all_prompts)
                print(f"[INFO] Loaded YOLOv8x-World with {len(all_prompts)} prompts")
                
                # Load Custom Model
                self.model_custom = YOLO('yolov11x_custom_trained_v2.pt')
                print(f"[INFO] Loaded YOLOv11x-Custom")
                
                # Set initial model
                self.detection_model = self.model_custom
            else:
                model_path = self.selected_model.get()
                self.detection_model = YOLO(model_path)
                
                # Set classes ONLY for YOLOv8-World models (open-vocabulary)
                # Standard YOLO models (like YOLOv11x) and custom models don't support set_classes()
                if "world" in model_path.lower() and "custom" not in model_path.lower():
                    all_prompts = []
                    for prompts in INSPECTION_ELEMENTS_BASE.values():
                        all_prompts.extend(prompts)
                    self.detection_model.set_classes(all_prompts)
                    print(f"[INFO] YOLOv8x-World: Using open-vocabulary detection with {len(all_prompts)} prompts")
                elif "grayscale" in model_path.lower():
                    print(f"[INFO] Grayscale Model: Using trained classes (fluorescent tube, fire extinguisher, outlet)")
                elif "custom" in model_path.lower():
                    print(f"[INFO] Custom Model: Using native trained classes")
                else:
                    print(f"[INFO] Standard YOLO Model: Using pre-trained COCO classes")
            
            self.camera_active = True
            self.camera_button.config(text="STOP", bg='#e74c3c')
            self.update_status("Video running", '#27ae60')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            self.update_status(f"Video error: {str(e)}", '#e74c3c')
            
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_active = False
        
        # Wait for camera thread to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
            # Give the system time to release the camera device
            time.sleep(0.5)
        
        self.camera_button.config(text="START", bg='#27ae60')
        self.camera_label.config(image='', text="Video stopped")
        self.update_status("Video stopped", '#95a5a6')
        
        # Clear detections
        self.detection_listbox.delete(0, tk.END)
        self.current_detections = []
        
    def camera_loop(self):
        """Main camera processing loop"""
        device = 0 if torch.cuda.is_available() else 'cpu'
        frame_count = 0
        
        try:
            while self.camera_active:
                frame_count += 1
                # Check if paused
                if self.paused:
                    time.sleep(0.1)
                    continue

                if not self.camera or not self.camera.isOpened():
                    break
                
                # Handle seeking
                if self.seek_frame is not None:
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                    self.seek_frame = None
                
                # Update slider if not dragging
                if not self.user_is_seeking:
                    current_pos = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
                    self.progress_slider.set(current_pos)
                    
                ret, frame = self.camera.read()
                if not ret:
                    # Loop video
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.camera.read()
                    if not ret:
                        break
                
                # Resize to 512px max dimension for speed
                h, w = frame.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # Only convert to grayscale if using grayscale model
                model_path = self.selected_model.get()
                if "grayscale" in model_path.lower():
                    # Convert to grayscale (Black & White) for grayscale models
                    # We convert to GRAY then back to BGR to keep 3 channels for compatibility
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Run detection
                if self.dual_model_mode.get():
                    if frame_count % 2 == 0:
                        self.detection_model = self.model_world
                    else:
                        self.detection_model = self.model_custom
                
                results = self.detection_model.predict(source=frame, conf=0.30, verbose=False, device=device)
                
                # Process detections
                detections = []
                
                # Clean up old inspected objects (remove if not seen for > 5 seconds)
                current_time = time.time()
                self.inspected_objects = [obj for obj in self.inspected_objects if current_time - obj['last_seen'] < 5.0]

                if results and len(results) > 0:
                    result = results[0]
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        current_box = [float(x1), float(y1), float(x2), float(y2)]
                        
                        # CHECK TRACKING: Compare with inspected objects
                        status = "N/A"
                        status_color = (0, 255, 255) # Yellow for N/A
                        
                        # Find best match among inspected objects
                        best_iou = 0
                        matched_obj_index = -1
                        
                        for i, obj in enumerate(self.inspected_objects):
                            # Only match if class is the same
                            if obj['class'] == class_name:
                                iou = self.calculate_iou(current_box, obj['box'])
                                if iou > best_iou:
                                    best_iou = iou
                                    matched_obj_index = i
                        
                        # If overlap is significant (> 0.3), inherit status
                        if best_iou > 0.3 and matched_obj_index != -1:
                            matched_obj = self.inspected_objects[matched_obj_index]
                            status = matched_obj['status']
                            
                            # Update the inspected object's position to track it
                            self.inspected_objects[matched_obj_index]['box'] = current_box
                            self.inspected_objects[matched_obj_index]['last_seen'] = current_time
                            
                            if status == "OK":
                                status_color = (0, 255, 0) # Green
                            elif status == "FAULTY":
                                status_color = (0, 0, 255) # Red
                        
                        detections.append({
                            "class_name": class_name,
                            "confidence": float(confidence),
                            "box": current_box,
                            "status": status # Add status to detection object
                        })
                        
                        # Draw on frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), status_color, 2)
                        
                        # Label with Status
                        label = f"{class_name} [{status}]"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Update detection list
                self.current_detections = detections
                self.update_detection_list(detections)
                
                # Convert frame to PhotoImage and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                # Larger feed size - 960x720 pixels
                # Use NEAREST for faster playback speed
                img.thumbnail((960, 720), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(img)
                
                self.camera_label.config(image=photo, text='')
                self.camera_label.image = photo

                time.sleep(0.01)  # Faster playback
                
        except Exception as e:
            print(f"[ERROR] Camera loop error: {e}")
        finally:
            # Ensure camera is released when loop exits
            if self.camera:
                self.camera.release()
                self.camera = None

    def update_detection_list(self, detections):
        """Update the detection listbox and auto-select the best detection"""
        self.detection_listbox.delete(0, tk.END)
        
        best_idx = -1
        max_conf = -1.0
        
        for i, det in enumerate(detections):
            # Show status in listbox too
            status_str = f" [{det.get('status', 'N/A')}]"
            self.detection_listbox.insert(
                tk.END, 
                f"{i+1}. {det['class_name']} ({det['confidence']:.2f}){status_str}"
            )
            
            # Track best detection
            if det['confidence'] > max_conf:
                max_conf = det['confidence']
                best_idx = i
        
        # Auto-select the highest confidence detection
        if best_idx != -1:
            self.detection_listbox.selection_set(best_idx)
            self.detection_listbox.see(best_idx)
            # Update the cropped view for the best detection
            self.update_cropped_view(best_idx)
            
    def update_cropped_view(self, idx):
        """Update the cropped view for a specific detection index"""
        if idx >= len(self.current_detections):
            return
            
        detection = self.current_detections[idx]
        
        # Crop and display
        if self.current_frame is not None:
            box = detection["box"]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Add padding
            padding = 10
            h, w = self.current_frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped = self.current_frame[y1:y2, x1:x2]
            
            # Convert and display
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cropped_rgb)
            # Much larger cropped image size - fill the 600x450 frame
            # Use NEAREST for speed
            img.thumbnail((600, 450), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(img)
            
            self.cropped_label.config(image=photo, text='')
            self.cropped_label.image = photo
            
            # Enable inspect button
            self.inspect_button.config(state=tk.NORMAL, bg='#e67e22')

    def on_detection_select(self, event):
        """Handle detection selection"""
        selection = self.detection_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        self.update_cropped_view(idx)
            
    def inspect_selected_detection(self):
        """Inspect the selected detection with Gemini"""
        print("[DEBUG] ========== INSPECT BUTTON CLICKED ==========")
        selection = self.detection_listbox.curselection()
        print(f"[DEBUG] Current selection: {selection}")
        if not selection:
            print("[DEBUG] No selection when inspect clicked!")
            return
        
        idx = selection[0]
        print(f"[DEBUG] Selected index: {idx}, total detections: {len(self.current_detections)}")
        if idx >= len(self.current_detections):
            print("[DEBUG] Index out of range!")
            return
        
        detection = self.current_detections[idx]
        print(f"[DEBUG] Will inspect: {detection}")
        
        # Run inspection in thread
        thread = threading.Thread(
            target=self.run_gemini_inspection, 
            args=(detection,), 
            daemon=True
        )
        thread.start()
        print("[DEBUG] Inspection thread started")
        
    def run_gemini_inspection(self, detection):
        """Run Gemini inspection on a detection"""
        try:
            print(f"[DEBUG] Starting inspection for: {detection['class_name']}")
            
            # Pause video playback
            self.paused = True
            
            # Disable button - thread-safe
            self.root.after(0, lambda: self.inspect_button.config(state=tk.DISABLED, bg='#7f8c8d'))
            self.update_status("Running Gemini inspection...", '#9b59b6')
            
            # Animate robot
            self.animate_robot_thinking()
            
            # Get category
            category = self.get_element_category(detection["class_name"])
            if not category:
                category = detection["class_name"]
            
            print(f"[DEBUG] Category: {category}")
            
            # Check if we have a current frame
            if self.current_frame is None:
                print("[ERROR] No current frame available!")
                self.add_result("[ERROR] No frame available for inspection")
                self.update_status("Error: No frame!", '#e74c3c')
                self.root.after(0, lambda: self.inspect_button.config(state=tk.NORMAL, bg='#e67e22'))
                return
            
            # Save cropped image
            box = detection["box"]
            x1, y1, x2, y2 = [int(v) for v in box]
            padding = 10
            h, w = self.current_frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped = self.current_frame[y1:y2, x1:x2]
            
            print(f"[DEBUG] Cropped image shape: {cropped.shape}")
            
            # Save temp file
            os.makedirs('gemini_results', exist_ok=True)
            temp_path = os.path.join('gemini_results', f'temp_inspection_{int(time.time())}.jpg')
            success = cv2.imwrite(temp_path, cropped)
            
            if not success:
                print(f"[ERROR] Failed to save cropped image to {temp_path}")
                self.add_result("[ERROR] Failed to save cropped image")
                self.update_status("Error: Save failed!", '#e74c3c')
                self.root.after(0, lambda: self.inspect_button.config(state=tk.NORMAL, bg='#e67e22'))
                return
            
            print(f"[DEBUG] Saved temp image to: {temp_path}")
            
            # Call Gemini
            print(f"[DEBUG] Calling Gemini API...")
            result = self.inspect_element_gemini(category, temp_path)
            
            print(f"[DEBUG] Gemini result: {result}")
            
            # Display result
            self.add_result(f"\n{'='*40}")
            self.add_result(f"INSPECTION: {detection['class_name']}")
            self.add_result(f"Confidence: {detection['confidence']:.2f}")
            self.add_result(f"{'='*40}")
            
            inspection_status = "N/A"
            
            r = result.lower().strip()
            if r.startswith("ok") or "good condition" in r:
                self.add_result(f"[OK] {result}", 'ok')
                inspection_status = "OK"
            elif r.startswith("faulty") or "defect" in r or "damage" in r:
                self.add_result(f"[FAULTY] {result}", 'faulty')
                inspection_status = "FAULTY"
            else:
                self.add_result(f"Result: {result}")
                inspection_status = "CHECK" # Ambiguous result
            
            # TRACKING: Save this inspection result
            # We store the box coordinates so we can match them in the next frame
            new_inspected_obj = {
                'class': detection['class_name'],
                'box': detection['box'],
                'status': inspection_status,
                'last_seen': time.time()
            }
            
            # Check if we should update an existing one or add new
            # (Simple logic: just append, the loop cleans up old ones)
            self.inspected_objects.append(new_inspected_obj)
            print(f"[TRACKING] Added inspected object: {detection['class_name']} -> {inspection_status}")
            
            self.update_status("Inspection complete!", '#27ae60')
            print("[DEBUG] Inspection complete!")
            
        except Exception as e:
            print(f"[ERROR] Exception in run_gemini_inspection: {e}")
            import traceback
            traceback.print_exc()
            
            self.add_result(f"[ERROR] {str(e)}")
            self.update_status("Inspection error!", '#e74c3c')
        finally:
            # Resume video playback
            self.paused = False
            
            # Re-enable button - thread-safe
            self.root.after(0, lambda: self.inspect_button.config(state=tk.NORMAL, bg='#e67e22'))
            print("[DEBUG] Button re-enabled")
            
    def get_element_category(self, detected_label):
        """Map a detected label back to its category"""
        detected_label = detected_label.lower()
        
        for category, prompts in INSPECTION_ELEMENTS_BASE.items():
            if any(prompt.lower() in detected_label for prompt in prompts):
                return category
        
        for category, prompts in INSPECTION_ELEMENTS_CUSTOM.items():
            if any(prompt.lower() in detected_label for prompt in prompts):
                return category
                
        return None
        
    def inspect_element_gemini(self, element_name, image_path):
        """Use Gemini 2.5 Flash to inspect the condition of a detected element"""
        api_key = GEMINI_API_KEY

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = f"""You are an inspection robot. You must ONLY respond in one of these two formats:
1. 'OK: [element] in good condition' if no issues are found
2. 'FAULTY: [brief reason]' if you spot any defects

No other response format is allowed. No explanations or commentary.

Inspect this {element_name}. Is it in good condition or faulty? Remember to use ONLY the required response format.
do not be strict on dirtiness of the object, focus on structural or functional defects. You can mention the dirtiness
of the object, but do not classify it as faulty just because it is dirty. the fire exinguishers do not require a
pressure gauge."""

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
    
    def cleanup(self):
        """Clean up resources before closing"""
        print("[INFO] Cleaning up resources...")
        if self.camera_active:
            self.stop_camera()
        cv2.destroyAllWindows()
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        box format: [x1, y1, x2, y2]
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Compute the area of both AABBs
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def try_multiple_urls(self, base_url):
        """
        Try multiple URL formats for the given IP address
        Returns (success, working_url, test_frame) or (False, None, None)
        """
        # Extract IP from URL
        import re
        ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', base_url)
        if not ip_match:
            return False, None, None
        
        ip = ip_match.group(1)
        
        # Common URL patterns to try
        url_patterns = [
            # DroidCam
            f"http://{ip}:4747/video",
            f"http://{ip}:4747/mjpegfeed",
            # IP Webcam
            f"http://{ip}:8080/videofeed",
            f"http://{ip}:8080/video",
            # iVCam / EpocCam
            f"http://{ip}:8080/",
            # Generic
            f"http://{ip}:8081/video",
        ]
        
        # If the original URL is already in the list, try it first
        if base_url in url_patterns:
            url_patterns.remove(base_url)
            url_patterns.insert(0, base_url)
        else:
            url_patterns.insert(0, base_url)
        
        print(f"[AUTO-DETECT] Trying {len(url_patterns)} URL formats...")
        
        for url in url_patterns:
            print(f"[AUTO-DETECT] Testing: {url}")
            
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[AUTO-DETECT] ✅ SUCCESS: {url}")
                        return True, url, frame
                    cap.release()
            except:
                pass
        
        print(f"[AUTO-DETECT] ❌ None of the URL formats worked")
        return False, None, None
    
    def test_camera_connection(self):
        """Test camera connection without starting the full camera loop"""
        def _test():
            try:
                source = self.camera_source.get()
                original_source = source
                
                # Parse source
                try:
                    source = int(source)
                    print(f"[TEST] Testing webcam index: {source}")
                    is_network = False
                except ValueError:
                    print(f"[TEST] Testing URL/path: {source}")
                    is_network = isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp'))
                
                self.update_status("Testing connection...", '#f39c12')
                
                # For network streams, try auto-detection first
                if is_network:
                    self.update_status("Auto-detecting URL format...", '#f39c12')
                    success, working_url, frame = self.try_multiple_urls(source)
                    
                    if success:
                        self.update_status(f"✅ Found working URL: {working_url}", '#27ae60')
                        print(f"[TEST] SUCCESS: Auto-detected working URL")
                        print(f"[TEST] Use this URL: {working_url}")
                        print(f"[TEST] Frame shape: {frame.shape}")
                        
                        # Update the camera source field with the working URL
                        def _update_field():
                            if working_url != original_source:
                                self.camera_source.set(working_url)
                        self.root.after(0, _update_field)
                        
                        return
                    else:
                        self.update_status("❌ Auto-detection failed - trying manual...", '#f39c12')
                
                # Manual test (original logic)
                print(f"[TEST] Opening camera...")
                test_cam = cv2.VideoCapture(source)
                
                # For network streams, configure timeouts
                if is_network:
                    test_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    try:
                        test_cam.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                        test_cam.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    except:
                        pass
                
                if not test_cam.isOpened():
                    self.update_status("❌ Connection failed - camera did not open", '#e74c3c')
                    print(f"[TEST] FAILED: Camera did not open")
                    
                    if is_network:
                        print(f"[TEST] WiFi camera troubleshooting:")
                        print(f"  1. Check phone and PC are on same WiFi")
                        print(f"  2. Verify IP in camera app matches URL")
                        print(f"  3. Try opening in browser: {source}")
                        print(f"  4. Run: python test_wifi_camera.py <IP>")
                        print(f"  5. See: WIFI_CAMERA_TROUBLESHOOTING.md")
                    
                    return
                
                print(f"[TEST] Camera opened")
                
                # Test reading frames
                print(f"[TEST] Reading test frame...")
                ret, frame = test_cam.read()
                
                test_cam.release()
                
                if not ret or frame is None:
                    self.update_status("❌ Connection failed - cannot read frames", '#e74c3c')
                    print(f"[TEST] FAILED: Cannot read frames")
                    
                    if is_network:
                        print(f"[TEST] Try different URL format (e.g., /videofeed, /mjpegfeed)")
                    
                    return
                
                # Success!
                self.update_status(f"✅ Connection OK - {frame.shape}", '#27ae60')
                print(f"[TEST] SUCCESS: Frame shape {frame.shape}")
                print(f"[TEST] Camera connection is working!")
                
            except Exception as e:
                self.update_status(f"❌ Test error: {str(e)}", '#e74c3c')
                print(f"[TEST] EXCEPTION: {type(e).__name__}: {e}")
        
        # Run test in thread to avoid blocking GUI
        threading.Thread(target=_test, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotInspectorGUI(root)
    
    # Register cleanup on window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
