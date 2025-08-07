import os
import time
import RPi.GPIO as GPIO
import cv2
import numpy as np
from pypylon import pylon
import datetime
import tifffile


# 이미지 저장 경로 설정
SAVE_PATH1 = "/"
SAVE_PATH2 = "/"

# GPIO 설정
GPIO.setmode(GPIO.BCM)
button_pin = 24
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 변수 설정
CAMERA_SIZE = [3840, 2160]  # Width, Height
EXPOSURE_TIME = 3000.0
GAIN = 10.0
MONO_MODE = "Mono8"  # Mono8 또는 Mono12

class BaslerCamera:
    def __init__(self, camera_size, exposure_time, gain, mono_mode):
        self.camera_size = camera_size
        self.exposure_time = exposure_time
        self.gain = gain
        self.mono_mode = mono_mode
        self.camera = None

    def initialize_camera(self):
        """카메라 초기화 및 설정"""
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            print(f"✅ Using device: {self.camera.GetDeviceInfo().GetModelName()}")

            # 카메라 설정
            self.camera.Width.SetValue(self.camera_size[0])
            self.camera.Height.SetValue(self.camera_size[1])
            self.camera.PixelFormat.SetValue(self.mono_mode)
            self.camera.ExposureTime.SetValue(self.exposure_time)
            self.camera.Gain.SetValue(self.gain)

            # 캡처 시작
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            self.camera = None


    def capture_photo(self):
        """이미지 캡처 및 저장 (continuous grabbing 상태에서 RetrieveResult 사용)"""
        with camera_lock:
            if self.camera is None:
                print("❌ Error: Camera instance is None!")
                return

            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            except Exception as e:
                print(f"❌ Error during RetrieveResult: {e}")
                return

            if not grab_result.GrabSucceeded():
                print("❌ Failed to capture image: No grab result")
                return

            img_array = grab_result.Array
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            if self.mono_mode == "Mono12":
                img_array &= 0xFFF  # 12비트 데이터 유지
                tiff_path = os.path.join(SAVE_PATH, f"{timestamp}.tiff")
                tifffile.imwrite(tiff_path, img_array, dtype="uint16", compression=None)

                img_resized = cv2.resize((img_array / 4095.0 * 255).astype(np.uint8), (384, 216), interpolation=cv2.INTER_AREA)
                png_path = os.path.join(SAVE_PATH, f"{timestamp}.png")
                cv2.imwrite(png_path, img_resized)

                print(f"✅ Image saved as TIFF: {tiff_path}")
            else:
                bmp_path = os.path.join(SAVE_PATH, f"{timestamp}.bmp")
                cv2.imwrite(bmp_path, img_array)

                img_resized = cv2.resize(img_array, (384, 216), interpolation=cv2.INTER_AREA)
                png_path = os.path.join(SAVE_PATH, f"{timestamp}.png")
                cv2.imwrite(png_path, img_resized)

                print(f"✅ Image saved as BMP: {bmp_path}")

            grab_result.Release()

    def release_resources(self):
        """카메라 및 리소스 해제"""
        if self.camera:
            self.camera.StopGrabbing()
            self.camera.Close()
        GPIO.cleanup()
        print("Camera and resources released successfully.")

def button_capture_loop(camera):
    """GPIO 버튼으로 이미지 캡처"""
    print("Waiting for button press...")
    try:
        while True:
            button_state = GPIO.input(button_pin)
            if button_state == GPIO.LOW:  # 버튼 눌림 감지
                print("Button Pressed. Capturing image...")
                time.sleep(1)
                camera.capture_photo()  # 사진 캡처
                print("Capture complete")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        camera.release_resources()

if __name__ == "__main__":
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    try:
        camera.initialize_camera()
        button_capture_loop(camera)
    except Exception as e:
        print(f"An error occurred: {e}")
        camera.release_resources()
