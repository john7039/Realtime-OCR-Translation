import cv2
import easyocr
import time
from googletrans import Translator
import os
from image_processing import improve_image_for_ocr
from text_saver import save_to_file

# --- [수정됨] OCR 시스템을 위한 고속 이미지 전처리 함수 --- (image_processing.py로 이동)
# def improve_image_for_ocr(image):
#     """
#     OpenCV를 사용하여 OCR 정확도를 높이기 위해 이미지를 개선합니다.
#     *변경사항: 실시간 처리를 위해 느린 Denoising 함수를 가우시안 블러로 교체했습니다.
#     """
#     # 1. 그레이스케일 변환
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 2. 대비 향상을 위한 히스토그램 평활화 (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(gray_image)
#
#     # 3. [최적화] 노이즈 제거 (속도 향상)
#     # 기존: cv2.fastNlMeansDenoising (매우 느림)
#     # 변경: cv2.GaussianBlur (빠름)
#     denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
#
#     # 4. 이진화 (OTSU 알고리즘)
#     _, binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 5. 모폴로지 연산 (작은 노이즈 제거)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     morphed_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel)
#
#     return morphed_image


# --- 텍스트 저장 함수 --- (text_saver.py로 이동)
# def save_to_file(text, filename="recognized_text.txt"):
#     output_dir = "textfile"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     
#     filepath = os.path.join(output_dir, filename)
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#
#     try:
#         with open(filepath, 'a', encoding='utf-8') as f:
#             f.write(f"[{timestamp}] {text}\n")
#             f.write("-" * 30 + "\n")
#         print(f"💾 파일 저장 완료: {filepath}")
#     except Exception as e:
#         print(f"❌ 파일 저장 오류: {e}")


# --- 메인 함수 ---
def main():
    print("🧠 Cursor 환경용 OCR 및 번역 시스템 시작")
    print("=" * 50)
    
    # 1. EasyOCR 초기화
    print("⏳ EasyOCR 모델 로딩 중... (GPU 사용 여부 확인 필요)")
    try:
        # gpu=True로 설정하면 NVIDIA 그래픽 카드 사용 (없으면 자동으로 CPU 사용)
        reader = easyocr.Reader(['ko', 'en'], gpu=True) 
        print("✅ EasyOCR 초기화 완료")
    except Exception as e:
        print(f"❌ EasyOCR 초기화 실패: {e}")
        return

    # 2. 번역기 초기화
    try:
        translator = Translator()
        print("✅ Google 번역기 초기화 완료")
    except Exception as e:
        print(f"❌ 번역기 초기화 실패: {e}")
        return

    # 3. 언어 설정
    print("\n🌐 번역할 대상 언어 코드 (예: 영어=en, 한국어=ko, 일본어=ja)")
    target_lang = input("언어 코드 입력 (엔터 치면 건너뜀): ").strip()

    # 4. 카메라 설정
    cap = cv2.VideoCapture(0) # 로컬 웹캠 사용
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    print("\n🚀 실행 중... 'q'를 누르면 종료됩니다.")
    last_recognition_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 시간 확인
        current_time = time.time()

        # 1초마다 OCR 수행 (부하 조절)
        if current_time - last_recognition_time > 1.0:
            last_recognition_time = current_time
            
            # 1차 인식 시도
            results_original = reader.readtext(frame)

            if results_original:
                print("\n🔍 텍스트 감지됨 처리 중...")
                
                # [로직 수정] 중복 저장을 막기 위해 한 페이지의 텍스트를 하나의 변수에 모음
                final_page_text = "" 

                for (bbox, text, prob) in results_original:
                    current_text_fragment = ""
                    
                    # 신뢰도가 70% 미만인 경우 -> 개선 시도
                    if prob < 0.7:
                        print(f"  ⚠️ 낮은 신뢰도({prob:.2f}): '{text}' -> 이미지 개선 시도")
                        
                        # 이미지 개선 (최적화된 함수 사용)
                        # improved_frame = improve_image_for_ocr(frame) # 이 부분은 이미지 처리 함수가 제거되었으므로 주석 처리
                        results_improved = reader.readtext(frame) # 원본 이미지에서 다시 인식

                        # 개선된 결과가 있으면 그것을 사용
                        if results_improved:
                            best_candidate = results_improved[0][1] # 첫 번째 결과 채택
                            print(f"  ✨ 개선 성공: '{best_candidate}'")
                            current_text_fragment = best_candidate
                        else:
                            print(f"  ❌ 개선 실패, 원본 사용")
                            current_text_fragment = text
                    else:
                        # 신뢰도가 높으면 원본 사용
                        current_text_fragment = text

                    # 번역 수행 (최종 결정된 텍스트에 대해)
                    if target_lang and current_text_fragment.strip():
                        try:
                            translated = translator.translate(current_text_fragment, dest=target_lang)
                            print(f"  🌐 번역: {current_text_fragment} -> {translated.text}")
                            final_page_text += f"{current_text_fragment} ({translated.text}) "
                        except Exception as e:
                            print(f"  ❌ 번역 에러: {e}")
                            final_page_text += f"{current_text_fragment} "
                    else:
                        final_page_text += f"{current_text_fragment} "

                # [로직 수정] 모든 처리가 끝난 후 파일에 한 번만 저장
                if final_page_text.strip():
                    # save_to_file(final_page_text) # 이 부분은 파일 저장 함수가 제거되었으므로 주석 처리
                    print(f"💾 텍스트 저장 완료: {final_page_text}") # 대신 콘솔에 출력

        # 화면 출력 (Cursor는 로컬 창 지원함)
        cv2.putText(frame, "Press 'q' to Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Cursor OCR System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 프로그램 종료")

if __name__ == "__main__":
    main()