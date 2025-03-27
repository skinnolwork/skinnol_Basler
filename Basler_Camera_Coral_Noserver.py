import os
import cv2
import time
import tifffile
from periphery import GPIO
from pypylon import pylon

# 이미지 저장 경로
SAVE_PATH = "/home/mendel/skinnol_Basler/images"

# GPIO 핀 설정 (GPIO138, physical pin 18)
button_pin = 138
button = GPIO(button_pin, "in")

# Basler 카메라 설정값 (예시)
CAMERA_SIZE = [3840, 2160]
EXPOSURE_TIME = 5000  # 예시값, 필요시 수정
GAIN = 0
MONO_MODE = "Mono12"

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
            self.camera.Width.SetValue(self.camera_size[0])
            self.camera.Height.SetValue(self.camera_size[1])
            self.camera.PixelFormat.SetValue(self.mono_mode)
            self.camera.ExposureTime.SetValue(self.exposure_time)
            self.camera.Gain.SetValue(self.gain)

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("✅ Camera initialized successfully!")

        except Exception as e:
            print(f"❌ Camera initialization error: {e}")

    def capture_image(self):
        """이미지 캡처 및 저장"""
        if self.camera is None:
            print("Camera not initialized.")
            return

        try:
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                img_array = grab_result.Array
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if self.mono_mode == "Mono12":
                    img_array &= 0xFFF
                    tiff_path = os.path.join(SAVE_PATH, f"{timestamp}.tiff")
                    tifffile.imwrite(tiff_path, img_array, dtype="uint16", compression=None)
                    print(f"✅ 이미지 저장 완료(TIFF): {tiff_path}")

                    # PNG 썸네일 저장
                    img_resized = cv2.resize((img_array/16).astype('uint8'), (384, 216), interpolation=cv2.INTER_AREA)
                    png_path = os.path.join(SAVE_PATH, f"{timestamp}.png")
                    cv2.imwrite(png_path, img_resized)
                    print(f"PNG Thumbnail saved: {png_path}")
                else:
                    # 다른 모드일 경우 bmp 저장
                    bmp_path = os.path.join(SAVE_PATH, f"{timestamp}.bmp")
                    cv2.imwrite(bmp_path, img_array)

                    img_resized = cv2.resize(img_array, (384, 216), interpolation=cv2.INTER_AREA)
                    png_path = os.path.join(SAVE_PATH, f"{timestamp}.png")
                    cv2.imwrite(png_path, img_resized)

                    print(f"Image saved: {png_path}")
            else:
                print("Grab failed.")
            grab_result.Release()
        except Exception as e:
            print(f"❌ Grab error: {e}")

    def release_resources(self):
        if self.camera is not None:
            self.camera.StopGrabbing()
            self.camera.Close()
            print("✅ Camera released.")

def button_capture_loop(camera, button_pin):
    """GPIO 버튼으로 이미지 캡처"""
    button = GPIO(button_pin, "in")
    print("Waiting for button press...")
    try:
        while True:
            if button.read() == False:  # 눌리면 LOW
                print("Button pressed! Capturing Image...")
                camera.capture_image()
                # 버튼이 계속 눌려있는 동안 반복 촬영 방지
                while button.read() == False:
                    time.sleep(0.05)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        button.close()
        camera.release_resources()

    print("GPIO cleanup done.")

if __name__ == "__main__":
    camera = BaslerCamera(CAMERA_SIZE, EXPOSURE_TIME, GAIN, MONO_MODE)
    camera.initialize_camera()
    button_capture_loop(camera, button_pin)
