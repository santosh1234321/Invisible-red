from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import threading
import queue
import sys
import os
import json

app = Flask(__name__)

# Global variables
camera = None
background = None
frame_queue = queue.Queue(maxsize=2)
is_running = False
app_stats = {
    'frames_processed': 0,
    'background_captures': 0,
    'start_time': time.time(),
    'current_resolution': '640x480'
}

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_initialized = False
        self.resolution = (640, 480)
        
    def find_working_camera(self):
        """Find a working camera configuration"""
        print("Searching for working camera...")
        
        configurations = [
            (0, cv2.CAP_DSHOW),
            (0, cv2.CAP_MSMF),
            (1, cv2.CAP_DSHOW),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
        ]
        
        for camera_idx, backend in configurations:
            try:
                print(f"Trying camera {camera_idx} with backend {backend}...")
                cap = cv2.VideoCapture(camera_idx, backend)
                
                if not cap.isOpened():
                    cap.release()
                    continue
                
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✅ Success! Camera {camera_idx} working")
                    
                    # Set properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Update stats
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    app_stats['current_resolution'] = f"{actual_w}x{actual_h}"
                    
                    self.cap = cap
                    self.is_initialized = True
                    return True
                    
                cap.release()
                
            except Exception as e:
                print(f"Camera {camera_idx} failed: {e}")
                continue
        
        return False
    
    def capture_background(self, num_frames=15):
        """Capture background with multiple attempts"""
        if not self.is_initialized or not self.cap:
            return None
        
        print("Capturing background...")
        time.sleep(2)
        
        backgrounds = []
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                backgrounds.append(frame)
            time.sleep(0.1)
        
        if not backgrounds:
            return None
        
        # Use median frame to reduce noise
        if len(backgrounds) > 5:
            background = np.median(backgrounds[-5:], axis=0).astype(np.uint8)
        else:
            background = backgrounds[-1]
            
        background = cv2.flip(background, 1)
        app_stats['background_captures'] += 1
        
        print(f"✅ Background captured: {background.shape}")
        return background
    
    def get_frame(self):
        """Get a single frame from camera"""
        if not self.is_initialized or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret and frame is not None and frame.size > 0:
            app_stats['frames_processed'] += 1
            return cv2.flip(frame, 1)
        return None
    
    def release(self):
        if self.cap:
            self.cap.release()
        self.is_initialized = False

def frame_producer():
    """Thread function to continuously capture frames"""
    global frame_queue, is_running, camera
    
    while is_running:
        frame = camera.get_frame()
        if frame is not None:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        
        time.sleep(0.033)  # ~30 FPS

def apply_invisibility_effect(frame, background_frame, sensitivity=0.6):
    """Apply invisibility cloak effect with adjustable sensitivity"""
    if background_frame is None:
        return frame
    
    # Resize background to match frame if needed
    if background_frame.shape != frame.shape:
        background_frame = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Dynamic red detection based on sensitivity
    lower_sat = int(50 + (100 - 50) * sensitivity)
    lower_val = int(50 + (100 - 50) * sensitivity)
    
    lower_red1 = np.array([0, lower_sat, lower_val])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, lower_sat, lower_val])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Enhance mask processing
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply effect with smooth blending
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    frame = frame.astype(float)
    background_frame = background_frame.astype(float)
    
    result = frame * (1 - mask_3d) + background_frame * mask_3d
    return result.astype(np.uint8)

def generate_frames():
    """Generate video stream"""
    global frame_queue, background
    
    while is_running:
        try:
            frame = frame_queue.get(timeout=1.0)
            
            if background is not None:
                frame = apply_invisibility_effect(frame, background)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except queue.Empty:
            # Error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Reconnecting...", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recapture_background')
def recapture_background():
    global background, camera
    
    if camera and camera.is_initialized:
        new_background = camera.capture_background()
        if new_background is not None:
            background = new_background
            return jsonify({"success": True, "message": "Background recaptured successfully!"})
        else:
            return jsonify({"success": False, "message": "Failed to capture background"})
    else:
        return jsonify({"success": False, "message": "Camera not initialized"})

@app.route('/stats')
def get_stats():
    runtime = time.time() - app_stats['start_time']
    fps = app_stats['frames_processed'] / max(runtime, 1)
    
    return jsonify({
        **app_stats,
        'runtime_seconds': round(runtime, 1),
        'fps': round(fps, 1),
        'camera_status': 'Connected' if (camera and camera.is_initialized) else 'Disconnected'
    })

def initialize_app():
    """Initialize the application"""
    global camera, background, is_running
    
    print("=== Invisibility Cloak App ===")
    
    camera = CameraManager()
    if not camera.find_working_camera():
        print("❌ Cannot initialize camera")
        return False
    
    background = camera.capture_background()
    if background is None:
        print("⚠️ Warning: Could not capture background")
    
    is_running = True
    frame_thread = threading.Thread(target=frame_producer, daemon=True)
    frame_thread.start()
    
    print("✅ App initialized successfully!")
    return True

if __name__ == '__main__':
    try:
        if initialize_app():
            app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
        else:
            print("App initialization failed. Exiting...")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        is_running = False
        if camera:
            camera.release()
        cv2.destroyAllWindows()
