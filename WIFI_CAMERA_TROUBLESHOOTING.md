# WiFi Camera Troubleshooting Guide

## Quick Diagnostic Steps

### Step 1: Run the WiFi Camera Test Tool

```bash
python test_wifi_camera.py 192.168.1.XXX
```

Replace `192.168.1.XXX` with your phone's IP address. This will test all common URL formats and tell you which one works.

### Step 2: Use the GUI Test Button

1. Enter your camera URL in the "Camera" field
2. Click the **TEST** button (blue button next to START)
3. Check the status message for results

---

## Common WiFi Camera Apps & Their URLs

### 📱 iVCam (Windows/Mac/Linux)
**Best for:** General purpose, good quality
- **Download:** App Store / Google Play
- **Default URL:** `http://YOUR_PHONE_IP:8080/video`
- **Alternative:** `http://YOUR_PHONE_IP:8080/`
- **Port:** 8080

**Setup:**
1. Install iVCam on phone
2. Open app → Note the IP address shown
3. Use URL: `http://IP_ADDRESS:8080/video`

---

### 📱 DroidCam (Android/iPhone)
**Best for:** Wireless and USB, works great on Linux
- **Download:** Google Play / App Store
- **Default URL:** `http://YOUR_PHONE_IP:4747/video`
- **MJPEG URL:** `http://YOUR_PHONE_IP:4747/mjpegfeed`
- **Port:** 4747

**Setup:**
1. Install DroidCam on phone
2. Start WiFi mode
3. Note the WiFi IP shown in app
4. Use URL: `http://IP_ADDRESS:4747/video`

---

### 📱 IP Webcam (Android)
**Best for:** Highly configurable, many features
- **Download:** Google Play
- **Video Feed:** `http://YOUR_PHONE_IP:8080/videofeed`
- **Video:** `http://YOUR_PHONE_IP:8080/video`
- **JPEG Stream:** `http://YOUR_PHONE_IP:8080/shot.jpg`
- **Port:** 8080 (configurable)

**Setup:**
1. Install IP Webcam
2. Scroll down and tap "Start server"
3. Note the URL shown at bottom of screen
4. Try both `/video` and `/videofeed` endpoints

---

### 📱 EpocCam (iPhone/iPad)
**Best for:** iOS users, high quality
- **Download:** App Store
- **Default URL:** `http://YOUR_PHONE_IP:8080/video`
- **Port:** 8080

**Setup:**
1. Install EpocCam on iPhone
2. Open app
3. Note IP address in app
4. Use URL: `http://IP_ADDRESS:8080/video`

---

## Troubleshooting Steps

### ❌ "Failed to open camera"

**Check 1: Same WiFi Network**
- Phone and computer MUST be on the same WiFi network
- Not cellular data!
- Check WiFi name matches on both devices

**Check 2: IP Address**
- Get IP from camera app (usually displayed prominently)
- IP should be like: `192.168.X.X` or `10.0.X.X`
- If it starts with `172.`, that's also valid (some routers)

**Check 3: Test in Browser**
- Open `http://YOUR_PHONE_IP:8080/video` in Chrome/Firefox
- If you see video in browser → URL is correct
- If browser fails → network/app issue

**Check 4: Firewall**
- Temporarily disable firewall on computer
- Test if camera works
- If it works, add exception for the port

**Check 5: Router Settings**
- Some routers have "AP Isolation" or "WiFi Isolation"
- This blocks devices from talking to each other
- Disable it in router settings (usually under WiFi settings)

---

### ❌ "Camera opened but cannot read frames"

**Option 1: Try Different URL Format**
```python
# If /video doesn't work, try:
http://IP:8080/videofeed
http://IP:8080/
http://IP:4747/mjpegfeed
```

**Option 2: Check App Settings**
- Make sure "Server" or "Streaming" is ON in the app
- Check quality settings (lower quality = faster)
- Try different resolution in app settings

**Option 3: Network Speed**
- WiFi signal strength matters!
- Move phone closer to router
- Close other apps using network

---

### ❌ "Connection is slow/laggy"

1. **Lower the resolution** in camera app settings
2. **Reduce frame rate** in app settings
3. **Move closer** to WiFi router
4. **Use 5GHz WiFi** if available (faster)
5. **Close other apps** on phone
6. **Consider USB connection** for best performance

---

## Testing Checklist

Use this checklist to systematically test your connection:

- [ ] Phone and PC on same WiFi network
- [ ] Camera app is running and shows an IP address
- [ ] IP address is correct format (192.168.X.X or 10.0.X.X)
- [ ] Can open URL in web browser and see video
- [ ] Firewall is disabled or has exception for port
- [ ] Router AP Isolation is disabled
- [ ] Phone is not in power-saving mode
- [ ] Phone is not in sleep mode
- [ ] Tried alternative URL formats (/video, /videofeed, etc.)
- [ ] Ran `python test_wifi_camera.py <IP>` and got success

---

## Quick Command Reference

### Find Your IP Address

**On Phone:**
- Settings → WiFi → Tap your network → Look for IP address
- Or check camera app (usually displays it)

**Test in Browser:**
```
http://192.168.1.100:8080/video
```

**Test with Python Script:**
```bash
python test_wifi_camera.py 192.168.1.100
```

**Use in GUI:**
- Enter: `http://192.168.1.100:8080/video`
- Click: **TEST** button
- If success, click: **START**

---

## Alternative: USB Connection

If WiFi is not working, USB connection is more reliable:

### DroidCam USB (Android)
```bash
# Install DroidCam client on PC
# Connect phone via USB
# Use source: 0 or 1 (webcam index)
```

### iPhone USB
```bash
# Use iVCam or EpocCam
# Connect via USB
# App will auto-detect USB connection
# Use source: 0 or 1 (webcam index)
```

---

## Port Reference

| App | Default Port | URL Format |
|-----|--------------|------------|
| iVCam | 8080 | `http://IP:8080/video` |
| DroidCam | 4747 | `http://IP:4747/video` |
| IP Webcam | 8080 | `http://IP:8080/videofeed` |
| EpocCam | 8080 | `http://IP:8080/video` |

---

## Still Not Working?

1. **Restart everything:**
   - Close camera app on phone
   - Restart phone
   - Restart computer
   - Restart router if needed

2. **Try a different app:**
   - If DroidCam doesn't work, try IP Webcam
   - Each app has different strengths

3. **Use USB instead of WiFi:**
   - More reliable
   - No network issues
   - Better performance

4. **Check the GitHub issues:**
   - Search for your camera app + "OpenCV"
   - Others may have solved your exact issue

5. **Get detailed diagnostics:**
   ```bash
   python test_wifi_camera.py YOUR_IP 2>&1 | tee camera_test.log
   ```
   This saves all output to a file you can share

---

## Success Tips

✅ **Lower resolution = better performance**
- Set camera app to 720p or lower
- Higher FPS is better than higher resolution

✅ **WiFi signal matters**
- Keep phone close to router during testing
- Use 5GHz WiFi if available

✅ **Browser test is your friend**
- If it works in browser, it should work in the app
- Browser test is fastest way to verify URL

✅ **TEST button before START**
- Always click TEST first
- Saves time by catching issues early

---

## Example Working Configurations

### Configuration 1: IP Webcam (Android)
```
Phone IP: 192.168.1.105
URL: http://192.168.1.105:8080/videofeed
Resolution: 720p
FPS: 15
WiFi: 2.4GHz
Result: ✅ Works perfectly
```

### Configuration 2: DroidCam (Android)
```
Phone IP: 192.168.0.150
URL: http://192.168.0.150:4747/video
Resolution: 480p
FPS: 30
WiFi: 5GHz
Result: ✅ Works perfectly
```

### Configuration 3: iVCam (iPhone)
```
Phone IP: 10.0.0.45
URL: http://10.0.0.45:8080/video
Resolution: 1080p
FPS: 15
WiFi: 5GHz
Result: ✅ Works perfectly
```

---

Need more help? Check:
- `PHONE_CAMERA_SETUP.md` - Full setup instructions
- `test_wifi_camera.py` - Automated testing tool
- Camera app's built-in help/FAQ

