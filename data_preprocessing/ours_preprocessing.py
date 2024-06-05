import os
import shutil

source_directory = '/experiment_dataset/E03'
destination_directory = 'valid/E03'

for filename in os.listdir(source_directory):
    shutil.move(os.path.join(source_directory, filename), 
                os.path.join(destination_directory, filename))
    
image_extensions = ['jpg', 'png', 'jpeg']

subdirectories = [d for d in os.listdir() if os.path.isdir(d)]


for subdir in subdirectories:
    e01_count = sum(1 for f in os.listdir(os.path.join(subdir, 'E01')) if f.split('.')[-1].lower() in image_extensions)
    e02_count = sum(1 for f in os.listdir(os.path.join(subdir, 'E02')) if f.split('.')[-1].lower() in image_extensions)
    e03_count = sum(1 for f in os.listdir(os.path.join(subdir, 'E03')) if f.split('.')[-1].lower() in image_extensions)

    print(f"Directory: {subdir}")
    print(f"E01: {e01_count} images")
    print(f"E02: {e02_count} images")
    print(f"E03: {e03_count} images")
    print("-" * 30)

# 원본 디렉토리와 대상 디렉토리의 경로를 지정
source_directory = '/test/E02'  
destination_directory = 'valid/E02' 

desired_string = 'tmp'

for filename in os.listdir(source_directory):
    # 파일 이름에 원하는 문자열이 포함되어 있으면
    if desired_string in filename:
        # 해당 파일을 대상 디렉토리로 이동
        shutil.move(os.path.join(source_directory, filename), 
                    os.path.join(destination_directory, filename))
