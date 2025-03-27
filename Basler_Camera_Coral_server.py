import os
import time
import cv2
import threading
import tifffile
import numpy as np
from flask import Flask, Response, send_from_directory
from periphery import GPIO
from pypylon import pylon
import datetime

# Flask 앱 생성
app = Flask(__name__)

# 이미지 저장 경로 설정
SAVE_PATH = "/home/mendel/skinnol_Basler/images"

# GPIO 설정 (Coral Dev Board: GPIO138, physical pin 18)
button_pin = 138
button = GPIO(button_pin, "in")

# Basler 카메라 설정
CAMERA_SIZE = [3840, 2160]
EXPOSURE_TIME = 1000000.0
GAIN = 40.0
MONO_MODE = "Mono12"

# 카메라 접근 보호를 위한 Lock 생성
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
            print(f"❌ Camera initialization error: {e}")
            self.camera = None

    def capture_photo(self):
        """이미지 캡처 및 저장"""
        with camera_lock:
            if self.camera is None:
                print("❌ Camera instance is None!")
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
                img_array &= 0xFFF
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
        with camera_lock:
            if self.camera:
                self.camera.StopGrabbing()
                self.camera.Close()
        print("Camera and resources released successfully.")


# Flask 엔드포인트 추가
camera_instance = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)


def generate_frames():
    """카메라로부터 실시간 프레임을 생성"""
    while True:
        with camera_lock:
            if not camera_instance.camera or not camera_instance.camera.IsGrabbing():
                continue

            grab_result = camera_instance.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                img_array = grab_result.Array
                if camera_instance.mono_mode == "Mono12":
                    img_array &= 0xFFF

                img_resized = cv2.resize(img_array, (640, 360), interpolation=cv2.INTER_AREA)  # 해상도 축소
                img_colored = cv2.cvtColor((img_resized / 4095.0 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                ret, buffer = cv2.imencode('.jpg', img_colored, [cv2.IMWRITE_JPEG_QUALITY, 50])


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
    """버튼으로 이미지 캡처"""
    print("📷 Waiting for button press...")

    try:
        while True:
            if button.read() == False:  # 눌리면 LOW
                print("📷 Button Pressed. Capturing image...")
                time.sleep(0.2)  # 버튼 바운싱 방지
                camera.capture_photo()  # 사진 촬영
                print("✅ Capture complete")

                # 버튼이 계속 눌려있는 동안 반복 촬영 방지
                while button.read() == False:
                    time.sleep(0.05)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("❌ Program terminated by user.")
    finally:
        button.close()
        camera.release_resources()


if __name__ == "__main__":
    try:
        camera_instance.initialize_camera()

        # Flask 서버와 버튼 감지 루프를 동시에 실행
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False))
        flask_thread.daemon = True
        flask_thread.start()

        button_capture_loop(camera_instance)
    except Exception as e:
        print(f"An error occurred: {e}")
        camera_instance.release_resources()
