import os
import time

def save_to_file(text, filename="recognized_text.txt"):
    output_dir = "textfile"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {text}\n")
            f.write("-" * 30 + "\n")
        print(f"💾 파일 저장 완료: {filepath}")
    except Exception as e:
        print(f"❌ 파일 저장 오류: {e}")
