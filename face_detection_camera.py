import cv2

# 분류기
faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 카메라
video_capture = cv2.VideoCapture(0)

while True:
    # 카메라에서 이미지 읽어들이기
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 찾기
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # 찾은 얼굴에 표시하기
    for (x, y, w, h) in faces:
        # 얼굴부위에 사각형 표시하기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 이미지 보여주기
    cv2.imshow('video', frame)

    # 사용자 키입력(q) 대기
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캡 연결종료
video_capture.release()
cv2.destroyAllWindows()