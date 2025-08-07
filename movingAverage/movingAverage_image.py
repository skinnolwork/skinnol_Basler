import tifffile
import numpy as np
import os

def load_tiff(filepath):
    return tifffile.imread(filepath).astype(np.float64)

def save_tiff(filepath, data):
    tifffile.imwrite(
        filepath,
        np.clip(data, 0, 4095).astype(np.uint16),
        dtype="uint16",
        compression=None
    )

def apply_sma_image(img, window):
    h, w = img.shape
    sma_img = np.zeros_like(img)

    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - pad_left - 1

    for i in range(h):
        # 가장자리 값을 복제하여 패딩
        padded_row = np.pad(img[i, :], (pad_left, pad_right), mode='edge')
        sma_row = np.convolve(padded_row, kernel, mode='valid')
        sma_img[i, :] = sma_row

    return sma_img

if __name__ == "__main__":
    input_path = "original_Mono12_20250411_110738.tiff"
    output_path = "sma_result.tiff"
    window = 9

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_path}")

    img = load_tiff(input_path)
    sma_img = apply_sma_image(img, window)
    save_tiff(output_path, sma_img)

    print(f"저장 완료: {output_path}")
