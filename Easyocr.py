import cv2
import easyocr
import time
from googletrans import Translator
import numpy as np
import os

# --- OCR 시스템을 위한 이미지 전처리 함수 (논문의 핵심) ---
def improve_image_for_ocr(image):
    """
    OpenCV를 사용하여 OCR 정확도를 높이기 위해 이미지를 개선합니다.
    - 그레이스케일 변환 및 대비 향상
    - 노이즈 제거
    - 텍스트 선명화 (샤프닝)
    - 이진화 및 모폴로지 연산 추가
    """
    # 1. 그레이스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 대비 향상을 위한 히스토그램 평활화 (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # 3. 노이즈 제거
    denoised_image = cv2.fastNlMeansDenoising(enhanced_image, None, 30, 7, 21)

    # 4. 이진화 (글자와 배경을 명확하게 분리)
    # OTSU 알고리즘을 사용하여 최적의 임계값을 자동으로 찾습니다.
    _, binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 모폴로지 연산 (작은 노이즈 제거 및 글자 연결)
    # 모폴로지 연산을 위한 커널(구조 요소) 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 오프닝(Opening) 연산: 침식(Erosion) 후 팽창(Dilation)
    # 작은 노이즈(점)를 제거하고 글자를 부드럽게 만듭니다.
    morphed_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel)

    # 최종 개선된 이미지 반환
    return morphed_image


# --- 텍스트를 파일에 저장하는 함수 ---
def save_to_file(text, filename="recognized_text.txt"):
    """
    인식된 텍스트를 특정 폴더에 추가합니다.
    """
    output_dir = "textfile"

    # 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 폴더 '{output_dir}'를 생성했습니다.")

    # 폴더 경로와 파일명을 결합
    filepath = os.path.join(output_dir, filename)

    try:
        # 'a' 모드는 파일이 존재하면 내용을 추가하고, 없으면 새로 생성합니다.
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
            f.write("-" * 30 + "\n") # 페이지 구분을 위한 구분선
        print(f"✅ 텍스트가 '{filepath}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")


# --- OCR 시스템을 위한 메인 함수 ---
def main():
    """
    OpenCV와 EasyOCR을 사용하여 실시간 텍스트 인식 및 번역 시스템을 구축합니다.
    카메라를 통해 책의 페이지를 스캔하고 텍스트를 인식한 뒤 번역합니다.
    """
    print("🧠 EasyOCR 기반 텍스트 인식 및 번역 시스템 구축 시작")
    print("=" * 50)
    print("✅ EasyOCR 리더를 초기화 중입니다. 잠시 기다려주세요...")

    try:
        reader = easyocr.Reader(['ko', 'en'])
        print("✅ EasyOCR 리더 초기화 완료.")
    except Exception as e:
        print(f"❌ EasyOCR 초기화 중 오류 발생: {e}")
        return

    try:
        translator = Translator()
        print("✅ 구글 번역기 초기화 완료.")
    except Exception as e:
        print(f"❌ 번역기 초기화 중 오류 발생: {e}")
        return

    # --- 번역 언어 선택 기능 추가 ---
    print("\n🌐 번역할 언어 코드를 입력하세요. (예: 영어 'en', 일본어 'ja', 중국어 'zh-cn')")
    target_lang = input("언어 코드: ")
    if not target_lang:
        print("❌ 언어 코드가 입력되지 않아 번역을 건너뜁니다.")
        target_lang = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
        return

    print("\n🚀 실시간 텍스트 인식 시작. 'q'를 누르면 종료됩니다.")
    print("--------------------------------------------------")
    print("🔍 카메라 화면에 책의 페이지를 비춰주세요.")

    last_recognition_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 카메라 연결 상태를 확인하세요.")
            break

        current_time = time.time()
        if current_time - last_recognition_time > 1.0:
            last_recognition_time = current_time

            # --- 1차 인식 시도 ---
            results_original = reader.readtext(frame)

            if results_original:
                recognized_text = ""

                print(f"\n📖 1차 인식 결과:")
                for (bbox, text, prob) in results_original:
                    print(f"  - 원본: '{text}' (확률: {prob:.2f})")
                    recognized_text += text + " "

                    # --- 2차 이미지 개선 및 재인식 (신뢰도 70% 미만일 때) ---
                    if prob < 0.7:
                        print(f"    ⚠️ 신뢰도가 낮아 이미지 개선 후 재인식합니다...")

                        # OpenCV로 이미지 개선
                        improved_frame = improve_image_for_ocr(frame)

                        # 개선된 이미지로 재인식
                        results_improved = reader.readtext(improved_frame)

                        if results_improved:
                            for (bbox_imp, text_imp, prob_imp) in results_improved:
                                print(f"    ✨ 개선 후: '{text_imp}' (확률: {prob_imp:.2f})")
                                # 번역 로직은 개선된 텍스트에만 적용
                                if target_lang:
                                    try:
                                        translated = translator.translate(text_imp, dest=target_lang)
                                        print(f"    - 번역 ({target_lang}): {translated.text}")
                                        recognized_text += translated.text + f" ({target_lang} 번역) "
                                    except Exception as e:
                                        print(f"    - 번역 오류: {e}")
                        else:
                            print("    ❌ 개선 후에도 텍스트를 찾을 수 없습니다.")
                    else:
                        # 신뢰도가 높으면 바로 번역
                        if target_lang:
                            try:
                                translated = translator.translate(text, dest=target_lang)
                                print(f"    - 번역 ({target_lang}): {translated.text}")
                                recognized_text += translated.text + f" ({target_lang} 번역) "
                            except Exception as e:
                                print(f"    - 번역 오류: {e}")

                # 모든 텍스트를 모아서 파일에 저장
                if recognized_text:
                    save_to_file(recognized_text)
            else:
                print("\n🤔 텍스트를 찾을 수 없습니다. 초점을 맞추고 다시 시도하세요.")

        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Real-time Text Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 시스템 종료")

if __name__ == "__main__":
    main()