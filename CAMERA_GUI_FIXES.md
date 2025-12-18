# Camera GUI Fixes - December 2, 2025

## Issues Fixed

### 1. **Segmentation Fault / Camera Resource Busy**
**Problem:** The camera was experiencing "Device or resource busy" errors after repeated open/close attempts, leading to segmentation faults.

**Root Cause:**
- The camera was not being properly released before attempting to reopen
- The camera thread was still running when `camera.release()` was called
- No delay was given for the OS to free the camera device

**Solution:**
- Added thread synchronization: Wait for camera thread to finish before releasing camera
- Set `self.camera = None` after releasing to prevent double-release
- Added 0.5 second delay after `camera.release()` to give OS time to free the device
- Added try-except-finally block in camera loop to ensure cleanup on errors
- Added proper cleanup method and registered it with window close event

**Changes in `stop_camera()`:**
```python
def stop_camera(self):
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
    
    # ... rest of cleanup
```

**Changes in `camera_loop()`:**
```python
def camera_loop(self):
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    try:
        while self.camera_active:
            if not self.camera or not self.camera.isOpened():
                break
            # ... processing
    except Exception as e:
        print(f"[ERROR] Camera loop error: {e}")
    finally:
        # Ensure camera is released when loop exits
        if self.camera:
            self.camera.release()
            self.camera = None
```

**Added cleanup on window close:**
```python
def cleanup(self):
    """Clean up resources before closing"""
    print("[INFO] Cleaning up resources...")
    if self.camera_active:
        self.stop_camera()
    cv2.destroyAllWindows()

# In main:
def on_closing():
    app.cleanup()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
```

### 2. **Report Tab Not Updating After Inspection**
**Problem:** Despite thread-safe `add_result()` calls and debug output, the Report widget was not showing inspection results.

**Root Cause:**
- The `animate_robot_thinking()` function was calling `self.root.update()` from a background thread
- This violates Tkinter's thread-safety rules and can cause GUI freezes and update failures
- Direct GUI widget updates (button config) from background thread were also problematic

**Solution:**
- Completely rewrote `animate_robot_thinking()` to be thread-safe using `root.after()` callbacks
- Changed all direct button updates in `run_gemini_inspection()` to use `root.after()`
- All GUI updates are now properly scheduled on the main thread

**Changes in `animate_robot_thinking()`:**
```python
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
```

**Changes in `run_gemini_inspection()`:**
```python
# Before (NOT thread-safe):
self.inspect_button.config(state=tk.DISABLED, bg='#7f8c8d')

# After (thread-safe):
self.root.after(0, lambda: self.inspect_button.config(state=tk.DISABLED, bg='#7f8c8d'))
```

## Testing Instructions

1. **Camera Stability Test:**
   - Start the camera
   - Stop the camera
   - Repeat 5-10 times
   - Should NOT see "Device or resource busy" errors
   - Should NOT see segmentation faults

2. **Inspection Report Test:**
   - Start the camera
   - Wait for detections to appear
   - Click on a detection in the list
   - Click "INSPECT" button
   - Watch the Report section - should see:
     - "Running Gemini inspection..." status
     - Robot eyes animating
     - Inspection results appearing in Report widget
     - Debug messages in terminal showing all steps

3. **Window Close Test:**
   - Start the camera
   - Close the window (click X)
   - Should see "[INFO] Cleaning up resources..." in terminal
   - No crashes or hanging

## Thread Safety Checklist

All GUI updates from background threads now use `root.after()`:
- ✅ Status label updates (`update_status()`)
- ✅ Report text updates (`add_result()`)
- ✅ Button state changes (inspect button)
- ✅ Robot animation (`animate_robot_thinking()`)

## Key Principles Applied

1. **Never call Tkinter methods from background threads directly**
   - Always use `root.after(0, callback)` to schedule on main thread

2. **Proper resource cleanup**
   - Wait for threads to finish before releasing resources
   - Add delays for OS to release hardware resources
   - Use try-finally to ensure cleanup happens

3. **Defensive programming**
   - Check if camera is still open before reading
   - Handle exceptions gracefully
   - Set references to None after cleanup

## Expected Behavior

- Camera should start and stop reliably without resource conflicts
- Inspection results should appear in the Report widget immediately after Gemini processing
- All debug messages should appear in terminal showing the workflow
- No segmentation faults or GUI freezes
- Clean shutdown when window is closed
