import os
import time
import threading
import datetime
import tifffile
import cv2
import numpy as np
import RPi.GPIO as GPIO
from flask import Flask, Response, send_from_directory
from pypylon import pylon

# Flask 앱
app = Flask(__name__)
SAVE_PATH = "/home/skinnol/skinnol-raspberrypi/public/images"

# GPIO 설정
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 24
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 카메라 설정
CAMERA_SIZE = [3840, 2160]
EXPOSURE_TIME = 1000000.0
GAIN = 40.0
MONO_MODE = "Mono12"
camera_lock = threading.Lock()

class BaslerCamera:
    def __init__(self):
        self.camera = None

    def initialize(self):
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            print(f"✅ Connected: {self.camera.GetDeviceInfo().GetModelName()}")

            self.camera.Width.SetValue(CAMERA_SIZE[0])
            self.camera.Height.SetValue(CAMERA_SIZE[1])
            self.camera.PixelFormat.SetValue(MONO_MODE)
            self.camera.ExposureTime.SetValue(EXPOSURE_TIME)
            self.camera.Gain.SetValue(GAIN)

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            print("❌ Camera Init Error:", e)
            self.camera = None

    def capture(self):
        with camera_lock:
            if self.camera is None:
                return
            result = self.camera.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                img = result.Array
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                if MONO_MODE == "Mono12":
                    img &= 0xFFF
                    tifffile.imwrite(os.path.join(SAVE_PATH, f"{timestamp}.tiff"), img, dtype='uint16')
                    img_small = cv2.resize((img / 4095.0 * 255).astype(np.uint8), (384, 216))
                    cv2.imwrite(os.path.join(SAVE_PATH, f"{timestamp}.png"), img_small)
                else:
                    bmp_path = os.path.join(SAVE_PATH, f"{timestamp}.bmp")
                    cv2.imwrite(bmp_path, img)

                print(f"✅ Saved: {timestamp}")
            result.Release()

    def get_frame(self):
        with camera_lock:
            if not self.camera.IsGrabbing():
                return None
            result = self.camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                frame = result.Array
                if MONO_MODE == "Mono12":
                    frame &= 0xFFF
                    frame = (frame / 4095.0 * 255).astype(np.uint8)
                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (640, 480))
                result.Release()
                return img
            result.Release()
            return None

    def close(self):
        with camera_lock:
            if self.camera:
                self.camera.StopGrabbing()
                self.camera.Close()
        GPIO.cleanup()
        print("✅ Camera Closed")


camera = BaslerCamera()

def generate_stream():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '<h1>📷 Live Camera</h1><img src="/video_feed" width="640" height="480">'

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory(SAVE_PATH, filename)

def watch_button():
    last = GPIO.HIGH
    while True:
        current = GPIO.input(BUTTON_PIN)
        if current == GPIO.LOW and last == GPIO.HIGH:
            print("📷 Button pressed")
            camera.capture()
            time.sleep(0.2)
        last = current
        time.sleep(0.1)

if __name__ == '__main__':
    try:
        camera.initialize()
        t = threading.Thread(target=watch_button)
        t.start()
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except Exception as e:
        print("❌ Error:", e)
        camera.close()
