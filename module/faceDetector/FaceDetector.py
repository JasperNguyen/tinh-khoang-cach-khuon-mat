import cv2
import os
import numpy as np
from . import conf

class FaceDetector():
    def __init__(self) -> None:
        self.faceNet = None # Face network model
        self.minAccuracy = 0.2 # Độ chính xác tối thiểu

    def setMinAccuracy(self, val):
        self.minAccuracy = val

    # Trả về đường dẫn tất cả các file có phần mở rộng nằm trong list phần mở rộng
    def _getFilePathFromFileNameExtension(self, dirPath:str, listExtension:tuple) -> dict:
        filePath = {}
        
        for fileName in os.listdir(dirPath):
            # Lấy phần mở rộng của tệp
            fileNameExtension = fileName.split('.')[-1].lower()

            if fileNameExtension in listExtension:
                filePath[fileNameExtension] = os.path.join(dirPath, fileName)
            
            if listExtension in filePath:
                break
        
        return filePath
    

    
    # Load face network model từ directory chứa caffe model và tệp mô tả kiến trúc mạng
    # Trả về exception khi xảy ra lỗi
    # Thông báo khi thành công
    def loadFaceNetFromDir(self, dirCaffeDNN:str=conf.FACE_CAFFE_DNN_DIR):
        
        # Get file path
        listExtension = ('prototxt', 'caffemodel')
        print(f'[i] Đang tìm file {str(listExtension)} trong thư mục: ')
        print(f'          "{dirCaffeDNN}" ')

        filePath = self._getFilePathFromFileNameExtension(dirCaffeDNN, listExtension)
        if listExtension != tuple(filePath.keys()):
            print(f'[ERROR]: Không tìm thấy file {str(listExtension)} trong thư mục: "{dirCaffeDNN}" ')
            exit()
        print(f'[=>] Lấy danh sách các file thành công !')

        # Load model
        try:
            print('[i] Đang thử đọc network model từ directory chứa caffe model và tệp mô tả kiến trúc mạng...')
            self.faceNet = cv2.dnn.readNetFromCaffe(filePath['prototxt'], filePath['caffemodel'])
            print('[=>] Đọc network model từ directory chứa caffe model và tệp mô tả kiến trúc mạng thành công !')
        except:
            print(f'[ERROR]: Không thể đọc network model từ thư mục: {str(filePath)} !')
            exit()
        
        

    # Trả về danh sách kết quả
    # (
    #       (
    #           {độ chính xác},
    #           ({startX}, {startY}, {endX}, {endY})
    #       ),
    # )
    def detect(self, img:np.ndarray) -> tuple:
        #Init
        result:list = []
        (h, w) = img.shape[:2]

        self.faceNet.setInput(
            cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )
        )

        detections:np.ndarray = self.faceNet.forward()[0, 0]

        for item in detections:
            acccuracy = item[2]
            if acccuracy < self.minAccuracy:
                continue
            
            # Xu ly box
            box = item[3:7]
            box = box*np.array([w,h, w,h])

            startX, startY, endX, endY = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            box = (startX, startY, endX, endY)
            result.append((acccuracy, box))
        #for
        return tuple(result)
    #def

    def detectFaces(self, img:np.ndarray, minSize = 50, minScale =16/9, border=10):
        detectResult = self.detect(img)
        faces = []

        minSize = 50
        minScale = 2

        for accuracy, box in detectResult:
            startX, startY, endX, endY = box
            w, h = (endX - startX), (endY - startY)

            #region check
            if w < minSize or h < minSize:
                continue

            if w/h > minSize or h/w > minScale:
                continue
            #endregion check

            #region Xu ly border
            if startX - border >= 0:
                startX -= border

            if startY - border >= 0:
                startY -= border

            if endX + border < w:
                endX += border
            
            if endY + border < h:
                endY += border
            #endregion Xu ly border

            aFace = (accuracy, img[startY:endY, startX:endX])
            faces.append(aFace)
        
        return faces
            

# ########################################################################################
# ## TEST

# import cv2
# import time

# videoCapture = cv2.VideoCapture(0)
# time.sleep(1)

# faceDetector = FaceDetector()
# faceDetector.loadFaceNetFromDir()


# while True:
#     success, frame = videoCapture.read()

#     detectResult = faceDetector.detect(frame)
#     if len(detectResult) > 0:
#         for accuracy, box in detectResult:
#             startX, startY, endX, endY = box
#             frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (255,255,0), 1)

#     cv2.imshow(winname='TEST FaceDetector', mat=frame)
#     if cv2.waitKey(1) == ord('q'):
#         break


# ########################################################################################
# ## TEST DETECT FACES

# import cv2
# import time

# videoCapture = cv2.VideoCapture(0)
# time.sleep(1)

# faceDetector = FaceDetector()
# faceDetector.loadFaceNetFromDir()


# isTrue = True

# while isTrue:
#     success, frame = videoCapture.read()
#     faces = faceDetector.detectFaces(frame)

#     if len(faces) > 0:
#         for i, (accuracy, aFace) in enumerate(faces):
#             cv2.imshow(winname=f'TEST FaceDetector {i}', mat=aFace)
#             if cv2.waitKey(0) == ord('q'):
#                 isTrue = False
#                 break