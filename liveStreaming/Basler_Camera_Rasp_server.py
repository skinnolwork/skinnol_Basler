import os
import time
import RPi.GPIO as GPIO
import cv2
import numpy as np
import threading  # 🔹 스레드 동기화를 위한 Lock 추가
from pypylon import pylon
import datetime
import tifffile
from flask import Flask, Response, send_from_directory

# Flask 앱 생성
app = Flask(__name__)

# 이미지 저장 경로 설정
SAVE_PATH = "/home/skinnol/skinnol-raspberrypi/public/images"

# GPIO 설정
GPIO.setmode(GPIO.BCM)
button_pin = 24
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 변수 설정
CAMERA_SIZE = [3840, 2160]  # Width, Height
EXPOSURE_TIME = 1000000.0
GAIN = 40.0
MONO_MODE = "Mono12"  # Mono8 또는 Mono12

# 🔹 카메라 접근을 보호하기 위한 Lock 생성
camera_lock = threading.Lock()


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

    def display_camera_feed(self, img_array):
        if self.mono_mode == "Mono12":
            # 이미 video_feed에서 마스킹을 했더라도
            # 여기서 다시 12비트 데이터를 8비트로 변환합니다.
            img_array = ((img_array / 4095) * 255).astype(np.uint8)
        img_colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        return img_colored


    def release_resources(self):
        """카메라 및 리소스 해제"""
        with camera_lock:
            if self.camera:
                self.camera.StopGrabbing()
                self.camera.Close()
        GPIO.cleanup()
        print("Camera and resources released successfully.")


# Flask 엔드포인트 추가
camera_instance = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)


def generate_frames():
    """카메라로부터 실시간 프레임을 생성"""
    while True:
        with camera_lock:  # 🔹 카메라가 사용 중일 때 다른 스레드가 접근 못하게 보호
            if not camera_instance.camera.IsGrabbing():
                continue

            grab_result = camera_instance.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                img_array = grab_result.Array
                if camera_instance.mono_mode == "Mono12":
                    img_array &= 0xFFF

                processed_img = camera_instance.display_camera_feed(img_array)
                ret, buffer = cv2.imencode('.png', processed_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
            grab_result.Release()


@app.route('/video_feed')
def video_feed():
    """실시간 스트리밍 엔드포인트"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static/<filename>')
def serve_file(filename):
    """저장된 이미지를 제공"""
    return send_from_directory(SAVE_PATH, filename)


def button_capture_loop(camera):
    """GPIO 버튼으로 이미지 캡처 (버튼이 눌렸다가 떼질 때만 촬영)"""
    print("📷 Waiting for button press...")
    last_button_state = GPIO.HIGH  # 초기 상태는 버튼이 안 눌린 상태

    try:
        while True:
            button_state = GPIO.input(button_pin)

            if camera.camera is None:  # 🔹 카메라가 None이면 재연결 시도
                print("⚠ Camera instance lost, reinitializing...")
                camera.initialize_camera()
                time.sleep(1)
                continue

            # 버튼이 LOW(눌림) 상태로 변경되었을 때 촬영 (떼기 전까지는 재촬영 안 함)
            if button_state == GPIO.LOW and last_button_state == GPIO.HIGH:
                print("📷 Button Pressed. Capturing image...")
                time.sleep(0.2)  # 버튼 Bounce 방지
                camera.capture_photo()  # 사진 촬영
                print("✅ Capture complete")

            last_button_state = button_state  # 버튼 상태 업데이트
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("❌ Program terminated by user.")
    finally:
        camera.release_resources()



if __name__ == "__main__":
    try:
        camera_instance.initialize_camera()
        # Flask 서버와 버튼 캡처 루프 동시에 실행
        from threading import Thread
        flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False))
        flask_thread.start()
        button_capture_loop(camera_instance)
    except Exception as e:
        print(f"An error occurred: {e}")
        camera_instance.release_resources()
