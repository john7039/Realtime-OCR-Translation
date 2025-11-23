import cv2
import numpy as np

def improve_image_for_ocr(image):
    """
    OpenCV를 사용하여 OCR 정확도를 높이기 위해 이미지를 개선합니다.
    *변경사항: 실시간 처리를 위해 느린 Denoising 함수를 가우시안 블러로 교체했습니다.
    """
    # 1. 그레이스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 대비 향상을 위한 히스토그램 평활화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # 3. [최적화] 노이즈 제거 (속도 향상)
    # 기존: cv2.fastNlMeansDenoising (매우 느림)
    # 변경: cv2.GaussianBlur (빠름)
    denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    # 4. 이진화 (OTSU 알고리즘)
    _, binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 모폴로지 연산 (작은 노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morphed_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel)

    return morphed_image
