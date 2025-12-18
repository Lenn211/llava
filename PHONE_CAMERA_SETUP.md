# 📱 Connecting Your Phone Camera to the Robot Inspector

## Overview
The new camera-enabled GUI (`main_gemini_gui_camera.py`) supports:
- ✅ Real-time object detection from live camera feed
- ✅ Phone camera connection via network streaming
- ✅ Webcam support
- ✅ Click-to-inspect: Select any detected object and press "Inspect" to run Gemini analysis

---

## Method 1: Using IP Webcam (Android) - RECOMMENDED

### Step 1: Install IP Webcam App
1. On your Android phone, install **"IP Webcam"** from Google Play Store
2. It's free and works great for streaming

### Step 2: Start the Server
1. Open IP Webcam app
2. Scroll down and tap **"Start Server"**
3. The app will show you a URL like: `http://192.168.1.100:8080`

### Step 3: Get the Stream URL
The video stream URL format is:
```
http://YOUR_PHONE_IP:8080/video
```

For example:
```
http://192.168.1.100:8080/video
```

### Step 4: Enter in GUI
1. Run the program: `python main_gemini_gui_camera.py`
2. In the "Camera Source" field, enter: `http://192.168.1.100:8080/video`
3. Click **"START CAMERA"**
4. Your phone's camera feed will appear with real-time detection!

---

## Method 2: Using DroidCam (Android & iOS)

### Step 1: Install DroidCam
- **Android**: Install "DroidCam" from Google Play
- **iOS**: Install "DroidCam" from App Store
- **Computer**: Install DroidCam client from https://www.dev47apps.com/

### Step 2: Connect
1. Make sure phone and computer are on same WiFi
2. Open DroidCam on your phone
3. Note the IP address and port (e.g., `192.168.1.100:4747`)
4. In the GUI, enter: `http://10.37.6.46:4747/video`

---

## Method 3: Using EpocCam (iOS) - For iPhone

### Step 1: Install EpocCam
- **iPhone**: Install "EpocCam" from App Store
- **Computer**: Install EpocCam drivers from https://www.kinoni.com/

### Step 2: Connect
1. Open EpocCam on iPhone
2. The app will connect automatically when on same network
3. In the GUI, EpocCam creates a virtual webcam
4. Use camera source: `1` or `2` (try different numbers to find it)

---

## Method 4: Using Your Computer's Webcam

### Simple - Just use:
- Camera Source: `0` (default webcam)
- Or `1`, `2`, etc. for additional cameras

---

## Detailed Usage Instructions

### Starting the Camera
1. **Enter Camera Source**
   - Webcam: `0`, `1`, `2`, etc.
   - Phone: `http://YOUR_IP:PORT/video`

2. **Click "START CAMERA"**
   - The model will load (takes a few seconds)
   - Real-time detection begins automatically
   - Green boxes appear around detected objects

### Inspecting Objects
1. **View Detections**
   - The right panel shows all detected objects
   - Format: "1. outlet (0.85)", "2. fire extinguisher (0.92)"

2. **Select an Object**
   - Click on any detection in the list
   - The cropped view will show the selected object

3. **Run Gemini Inspection**
   - Click **"INSPECT SELECTED"** button
   - Gemini will analyze the object
   - Results appear in the inspection report
   - Robot eyes animate while thinking!

### Stopping the Camera
- Click **"STOP CAMERA"** button
- Camera feed stops
- Detection list clears

---

## Network Requirements

### Same WiFi Network
Your phone and computer **must be on the same WiFi network**!

### Find Your Phone's IP Address

**Android:**
1. Settings → WiFi → Connected network → Advanced
2. Look for "IP address"

**iOS:**
1. Settings → WiFi → (i) button next to connected network
2. Look for "IP Address"

### Firewall
If connection fails:
- Disable firewall temporarily to test
- Or add exception for the streaming app

---

## Troubleshooting

### "Failed to open camera"
**Cause**: Invalid source or network issue

**Solutions**:
- Check IP address is correct
- Ensure phone and PC on same WiFi
- Try restarting the streaming app
- For webcam: try `0`, `1`, or `2`

### Camera lag or frozen
**Cause**: Network bandwidth or CPU

**Solutions**:
- Move closer to WiFi router
- Close other apps on phone
- Reduce detection confidence: Edit line in code `conf=0.30` to `conf=0.50`

### No detections showing
**Cause**: Model needs better view or objects not in training

**Solutions**:
- Point camera closer to objects
- Ensure good lighting
- Try switching between Base and Custom model
- Lower confidence threshold

### Gemini inspection fails
**Cause**: API key or network

**Solutions**:
- Check GEMINI_API_KEY is correct
- Ensure internet connection
- Check Gemini API quota

---

## Example URLs

### IP Webcam (Android)
```
http://192.168.1.100:8080/video
http://192.168.0.55:8080/video
http://10.0.0.123:8080/video
```

### DroidCam
```
http://192.168.1.100:4747/video
```

### RTSP streams (advanced)
```
rtsp://192.168.1.100:8554/video
```

---

## Performance Tips

### For Best Performance:
1. **Use WiFi 5GHz** (if available) instead of 2.4GHz
2. **Good Lighting** - Better lighting = better detection
3. **Stable Phone Position** - Use a phone stand/tripod
4. **Close Background Apps** on both phone and computer
5. **Lower Resolution** in streaming app if lagging

### Detection Speed:
- **With GPU**: ~30 FPS (smooth)
- **CPU only**: ~5-10 FPS (acceptable)

---

## Advanced: Custom Streaming

### If you have your own RTSP/HTTP stream:
```python
# In the Camera Source field, enter:
rtsp://username:password@192.168.1.100:554/stream
http://192.168.1.100:8080/stream.mjpg
```

### Using VLC to Stream (Desktop Camera to Network):
```bash
# On source computer (with camera):
vlc v4l2:///dev/video0 :sout='#transcode{vcodec=mp4v,vb=800,scale=1}:http{mux=ffmpeg{mux=flv},dst=:8080/stream}' :sout-keep

# In GUI, use:
http://SOURCE_COMPUTER_IP:8080/stream
```

---

## Code Features

### Real-time Detection
- Runs continuously at ~30 FPS
- Uses selected model (Base or Custom)
- Shows all detections in list

### On-Demand Inspection
- Gemini only runs when you click "Inspect"
- Saves API calls and bandwidth
- You control what gets inspected

### Multi-threading
- Camera runs in separate thread
- GUI stays responsive
- Inspection runs in background thread

---

## Testing Checklist

1. ✅ Install IP Webcam on phone
2. ✅ Start server, note IP address
3. ✅ Phone and PC on same WiFi
4. ✅ Run `python main_gemini_gui_camera.py`
5. ✅ Enter phone URL in Camera Source
6. ✅ Click START CAMERA
7. ✅ Point camera at outlet/fire extinguisher/light
8. ✅ See detections appear in list
9. ✅ Click on a detection
10. ✅ Click INSPECT SELECTED
11. ✅ See Gemini analysis results

---

## Next Steps

### Want to save snapshots?
Add a "Capture" button that saves the current frame + annotations

### Want to log inspections?
Export inspection results to CSV/JSON file

### Want to run batch inspections?
Automatically inspect all detections when you press a button

Let me know if you want any of these features added! 🚀
