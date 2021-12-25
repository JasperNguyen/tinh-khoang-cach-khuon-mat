import numpy as np
import cv2
import time
from module.faceDetector.FaceDetector import FaceDetector

CV_WINNAME = 'khoang-cach-khuon-mat'

IN = 39.3700787
FEET = 3.2808399
CM = 100

C = 120
B = 0.06
CB = C*B

# Đơn vị là m
def tinhKhoangCach(A: float):
    return (CB/A)*10




# 1m = 3.2808399 


#Aqua
BORDER_COLOR = (255,255,0)

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(1)


    faceDetector = FaceDetector()
    faceDetector.loadFaceNetFromDir('./module/faceDetector/face_caffe_dnn')
    faceDetector.setMinAccuracy(0.5)

    while(True):
        isSuccess, frame = cap.read()

        if not isSuccess:
            break

        
        # Lấy khuôn mặt
        detectResult = faceDetector.detect(frame)
        if len(detectResult) > 0:

            for accuracy, box in detectResult:
                startX, startY, endX, endY = box
                w = (endX - startX)

                khoangCach = round(tinhKhoangCach(w), 2)

                # draw
                frame = cv2.rectangle(
                    frame,
                    pt1=box[0:2],
                    pt2=box[2:4],
                    color=BORDER_COLOR,
                    thickness=1
                )

                frame = cv2.putText(
                    img=frame,
                    text=f'{round(khoangCach*FEET, 2)} feet',
                    org=(startX + 5, startY + 20),
                    fontFace=cv2.cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1,
                    color=BORDER_COLOR,
                    thickness=1
                )

                frame = cv2.putText(
                    img=frame,
                    text=f'{round(khoangCach*CM, 2)} cm',
                    org=(startX + 5, startY + 40),
                    fontFace=cv2.cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1,
                    color=BORDER_COLOR,
                    thickness=1
                )

                frame = cv2.putText(
                    img=frame,
                    text=f'{round(khoangCach*IN, 2)} inch',
                    org=(startX + 5, startY + 60),
                    fontFace=cv2.cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1,
                    color=BORDER_COLOR,
                    thickness=1
                )


        cv2.imshow(winname=CV_WINNAME, mat=frame)
        if cv2.waitKey(1) == ord('q'):
            break

    #while
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
