import cv2
import numpy as np
import os

def get_manipulated_images(image, save_folder):
    """
    Görüntüyü farklı ışık koşullarında manipüle eder ve kaydeder.
    Logaritmik dönüşüm kaldırıldı, sadece gamma düzeltmesi ve histogram eşitleme var.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Orijinal görüntüyü kaydet (BGR olarak kaydet)
    original_path = os.path.join(save_folder, "original.jpg")
    cv2.imwrite(original_path, image)

    # Gamma düzeltmesi
    for gamma in [0.5, 1.5]:
        gamma_corrected = np.power(image / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        gamma_path = os.path.join(save_folder, f"gamma_{gamma}.jpg")
        cv2.imwrite(gamma_path, gamma_corrected)

    # Histogram eşitleme
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    hist_eq_path = os.path.join(save_folder, "hist_eq.jpg")
    cv2.imwrite(hist_eq_path, hist_eq)

    return {
        "original": image,
        "gamma_0.5": cv2.imread(os.path.join(save_folder, "gamma_0.5.jpg")),
        "gamma_1.5": cv2.imread(os.path.join(save_folder, "gamma_1.5.jpg")),
        "hist_eq": cv2.imread(os.path.join(save_folder, "hist_eq.jpg")),
    }

def get_wb_images(img):
    """
    Görüntüye Gray World algoritması ile renk sabitliği uygular.
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result