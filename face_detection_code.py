"""
conda install -c conda-forge opencv
"""
import os
import cv2

# 분류기
faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 샘플 이미지
cwd = os.getcwd()
sample_image = os.path.join(cwd, 'sample_images/celebrities.jfif')

# 이미지 읽어들이기
image = cv2.imread(sample_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 찾기
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(20,20),
)

# 찾은 얼굴에 표시하기
for (x,y,w,h) in faces:
    # 얼굴 부위에 사각형 표시
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    # 얼굴 부위만 잘라내기
    roi=image[y:y+h, x:x+w]

# 이미지 보여주기
cv2.imshow('image', image)

# 얼굴부위만 보여주기
cv2.imshow('face', roi)

# 사용자 키입력(esc) 대기
cv2.waitKey(0)
cv2.destroyAllWindows()