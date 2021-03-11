## 머신러닝, 딥러닝 처리하기 ##

from tkinter import *
from tkinter.filedialog import *
from tkinter import filedialog
from tkinter.simpledialog import *
import cv2
import numpy
import numpy as np
import dlib
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import sys, subprocess

# subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'opencv-contrib-python'])

## 함수 선언부
def malloc(h, w, value=0) :
    retMemory = [ [ value for _ in range(w)]  for _ in range(h) ]
    return retMemory

def mallocNumpy(t, h, w) :
    retMemory = np.zeros((t, h, w), dtype = np.int16)
    return retMemory

def allocateOutMemory() : # 메모리 할당 공통 함수 / 처음 알고리즘 -> numpy 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    # outImage = []
    # for _ in range(RGB) :
    #     outImage.append(malloc(outH, outW))
    outImage = mallocNumpy(RGB, outH, outW)
    return outImage

def bufferFile() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    inH = outH;    inW = outW
    ## 버퍼 메모리 할당
    inImage = mallocNumpy(RGB, outH, outW)

    ### 진짜 영상처리 알고리즘 ###
    inImage = outImage

def undoImage() : # 임시메모리를 이용해 버퍼이미지(입력이미지)와 출력이미지를 바꿔줌
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    # 임시 메모리에 현재 출력이미지 저장
    tmpH = outH ; tmpW = outH
    tmpImage = mallocNumpy(RGB, outH, outW)
    tmpImage = outImage

    #입력이미지를 출력으로 (이전이미지를 현재 출력으로)저장
    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()
    outH = inH;    outW = inW
    outImage = inImage

    #임시메모리에 있던 출력이미지를 입력으로 저장
    inH = tmpH;    inW = tmpW
    inImage = []
    inImage = mallocNumpy(RGB, tmpH, tmpW)
    inImage = tmpImage

    print('undoImage success')
    displayImageColor()

def openFile() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    ## 파일 선택하기
    filename = askopenfilename(parent=window,
           filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))
    ## (중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)
    inH = cvInImage.shape[0]
    inW = cvInImage.shape[1]
    ## 입력이미지용 메모리 할당
    # inImage = []
    # for _ in range(RGB) :
    #     inImage.append(malloc(inH, inW))

    inImage = mallocNumpy(RGB, inH, inW)
    ## 파일 --> 메모리 로딩

    for i in range(inH):
        for k in range(inW):
            inImage[R][i][k] = cvInImage.item(i, k ,B)
            inImage[G][i][k] = cvInImage.item(i, k, G)
            inImage[B][i][k] = cvInImage.item(i, k, R)

    equalColor()

def saveImage() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == None or filename == "" :
        return

    saveCvPhoto = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH) :
        for k in range(outW) :
            tup = tuple(([outImage[B][i][k],outImage[G][i][k],outImage[R][i][k]]))
            saveCvPhoto[i,k] = tup

    saveFp = asksaveasfile(parent=window, mode='wb',defaultextension='.', filetypes=(("그림 파일", "*.png;*.jpg;*.bmp;*.tif"), ("모든 파일", "*.*")))

    if saveFp == '' or saveFp == None:
        return
    cv2.imwrite(saveFp.name, saveCvPhoto)

def displayImageColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    window.geometry(str(outW)+'x'+str(outH))
    if canvas != None :
        canvas.destroy()

    # 최대 화면 제한하기
    VX, VY = 512,  512 # 최대 화면 크기

    ## 크기가 512보다 크면, 최대 512로 보이기....
    if outH <= VY or outW <= VX:
        VX = outW
        VY = outH
        step = 1
    else:
        if outH > outW:
            step = outH / VY  # 1024/512 = 2
            VY = int(outH / outW * VX)
        else:
            step = outW / VX  # 1024/512 = 2
            VX = int(outW / outH * VY)

    window.geometry(str(int(VX * 1.2)) + 'x' + str(int(VY * 1.2)))
    canvas = Canvas(window, height=VY, width=VX)
    paper = PhotoImage(height=VY, width=VX)
    canvas.create_image(( VX // 2, VY // 2), image=paper, state='normal')

    # 메모리에서 처리한 후, 한방에 화면에 보이기 --> 완전 빠름
    rgbString =""
    for i in numpy.arange(0, outH, step) :
        tmpString = "" # 각 줄
        for k in numpy.arange(0, outW, step) :
            i = int(i) # i,k 값은 실수가 될 수없어서 정수로 변환
            k = int(k)
            r = outImage[R][i][k]
            g = outImage[G][i][k]
            b = outImage[B][i][k]
            tmpString += "#%02x%02x%02x " % (r, g, b)
        rgbString += '{' + tmpString + '} '
    paper.put(rgbString)
    canvas.pack(expand = 1, anchor = CENTER)
    status.configure(text='이미지정보:' + str(outH) + 'x' + str(outW)+'      '+filename)

#영상처리 함수
#### 화소점 처리 ####
def equalColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    # outImage = []
    # for _ in range(RGB) :
    #     outImage.append(malloc(outH, outW))
    #outImage = allocateOutMemory()

    #outImage = mallocNumpy(RGB,outH,outW)
    ### 진짜 영상처리 알고리즘 ###
    # for rgb in range(RGB):
    #     for i in range(inH):
    #         for k in range(inW):
    #             outImage[rgb][i][k] = inImage[rgb][i][k]

    outImage = inImage.copy()
    ########################
    displayImageColor()

### OpenCV 함수 부분 ###
def cvOut2outImage() : # openCV의 결과 --> outImage의 메모리에 넣기
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    #결과 메모리의 크기
    outH = cvOutImage.shape[0]
    outW = cvOutImage.shape[1]
    ## 입력이미지용 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))

    ## cvOut --> 메모리 할당
    for i in range(outH):
        for k in range(outW):
            if (cvOutImage.ndim == 2) : # 그레이 or 흑백 일때
                outImage[R][i][k] = cvOutImage.item(i, k)
                outImage[G][i][k] = cvOutImage.item(i, k)
                outImage[B][i][k] = cvOutImage.item(i, k)
            else :
                outImage[R][i][k] = cvOutImage.item(i, k ,B)
                outImage[G][i][k] = cvOutImage.item(i, k, G)
                outImage[B][i][k] = cvOutImage.item(i, k, R)

### 머신러닝 함수 부분 ###
def faceDetect_CV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    face_cascade = cv2.CascadeClassifier('harrcascade/haarcascade_frontalface_alt.xml')
    grey = cv2.cvtColor(cvInImage[:], cv2.COLOR_BGR2GRAY)

    ## 얼굴이 여러개면 여러개를 찾기.
    # 얼굴 위치 사각형 [ [x1, y1, w, h] , [x2, y2, w, h] ... ]

    fact_rects = face_cascade.detectMultiScale(grey, 1.1, 5)

    cvOutImage = cvInImage[:]
    for x, y, w, h in fact_rects:
        cv2.rectangle(cvOutImage, (x, y), (x + h, y + w), (0, 255, 0), 3)

    cvOut2outImage()
    displayImageColor()

def noseDetect_CV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    face_cascade = cv2.CascadeClassifier('harrcascade/haarcascade_mcs_nose.xml')
    grey = cv2.cvtColor(cvInImage[:], cv2.COLOR_BGR2GRAY)

    ## 얼굴이 여러개면 여러개를 찾기.
    # 얼굴 위치 사각형 [ [x1, y1, w, h] , [x2, y2, w, h] ... ]

    fact_rects = face_cascade.detectMultiScale(grey, 1.1, 5)

    cvOutImage = cvInImage[:]
    for x, y, w, h in fact_rects:
        cv2.rectangle(cvOutImage, (x, y), (x + h, y + w), (0, 255, 0), 3)

    cvOut2outImage()
    displayImageColor()

def earDetect_CV() : # openCV이용 얼굴인식
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    earLeft_cascade = cv2.CascadeClassifier('harrcascade/haarcascade_mcs_leftear.xml')
    earRight_cascade = cv2.CascadeClassifier('harrcascade/haarcascade_mcs_rightear.xml')
    grey = cv2.cvtColor(cvInImage[:], cv2.COLOR_BGR2GRAY)
    fact_rects1 = earLeft_cascade.detectMultiScale(grey, 1.1, 5)
    fact_rects2 = earRight_cascade.detectMultiScale(grey, 1.1, 5)
    # 얼굴찾기
    cvOutImage = cvInImage[:]

    for x, y, w, h in fact_rects1:
        cv2.rectangle(cvOutImage, (x, y), (x + h, y + w), (0, 255, 0), 3)
    for x, y, w, h in fact_rects2:
        cv2.rectangle(cvOutImage, (x, y), (x + h, y + w), (0, 255, 0), 3)

    cvOut2outImage()
    ########################
    displayImageColor()

def catFaceDetect_CV() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return

    face_cascade = cv2.CascadeClassifier('harrcascade/haarcascade_frontalcatface.xml')
    grey = cv2.cvtColor(cvInImage[:], cv2.COLOR_BGR2GRAY)

    ## 얼굴이 여러개면 여러개를 찾기.
    # 얼굴 위치 사각형 [ [x1, y1, w, h] , [x2, y2, w, h] ... ]

    fact_rects = face_cascade.detectMultiScale(grey, 1.1, 5)

    cvOutImage = cvInImage[:]
    for x, y, w, h in fact_rects:
        cv2.rectangle(cvOutImage, (x, y), (x + h, y + w), (0, 255, 0), 3)

    cvOut2outImage()
    displayImageColor()

def faceDetectMosaic_CV() : # openCV이용 얼굴인식
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    face_cascade = cv2.CascadeClassifier('harrcascade/haarcascade_frontalface_alt.xml')
    grey = cv2.cvtColor(cvInImage[:], cv2.COLOR_BGR2GRAY)
    fact_rects = face_cascade.detectMultiScale(grey, 1.1, 5)
    # 얼굴찾기
    cvOutImage = cvInImage[:]

    # 얼굴 인식 실행하기 --- (※3)
    mosaic_rate = 30
    face_list = face_cascade.detectMultiScale(grey,scaleFactor=1.1, minNeighbors=1,minSize=(100, 100))
    if len(face_list) == 0:
        print("no face")
        quit()
    # 확인한 부분에 모자이크 걸기 -- (※4)
    print(face_list)
    color = (0, 0, 255)
    for (x, y, w, h) in face_list:
        # 얼굴 부분 자르기 --- (※5)
        face_img = cvInImage[y:y + h, x:x + w]
        # 자른 이미지를 지정한 배율로 확대/축소하기 --- (※6)
        face_img = cv2.resize(face_img, (w // mosaic_rate, h // mosaic_rate))
        # 확대/축소한 그림을 원래 크기로 돌리기 --- (※7)
        face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
        # 원래 이미지에 붙이기 --- (※8)
        cvInImage[y:y + h, x:x + w] = face_img
    # 렌더링 결과를 파일에 출력

    cvOut2outImage()
    ########################
    displayImageColor()

### 딥러닝 함수 부분 ###
def ssdNet(image) :
    CONF_VALUE = 0.8 # 20% 인정
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image

def deepStopImage_CV() : # 이미지 사물인식
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    if filename == None:
        return
    ##### OpenCV 용 영상처리 ###
    cvOutImage = ssdNet(cvInImage)

    cvOut2outImage()
    ########################
    displayImageColor()

def deepMoveImage_CV() : # 동영상 사물인식
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    movieName =  askopenfilename(parent=window,
           filetypes=(('동영상 파일', '*.mp4;*.avi'), ('All File', '*.*')))
    s_factor = 0.5 # 화면 크기 비율(조절 가능)
    capture = cv2.VideoCapture(movieName)
    #capture = cv2.VideoCapture(0)
    frameCount = 0 # 처리할 프레임의 숫자 (자동증가)
    ##### OpenCV 용 영상처리 ###
    while True:
        ret, frame = capture.read()
        if not ret:  # 동영상을 읽기 실패
            break
        frameCount += 1
        if frameCount % 10 == 0 : # 숫자 조절 가능 (속도 문제)
            frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
            ## 1장짜리 SSD 딥러닝 ##
            retImage = ssdNet(frame)
            ####################
            cv2.imshow('Video', retImage)

        key = cv2.waitKey(1) # 화면 속도 조절
        if key == 27:  # esc 키
            break
        elif key == ord('c') or key == ord('C'):
            # 키보드가 아닌, 조건에 의해서 처리도 가능함...
            # 예로 사람이 3명이상 등장하면......  강아지가 나타나면...
            cvInImage = cvOutImage = retImage
            filename = movieName
            cvOut2outImage()
            displayImageColor()

    capture.release()

def deepMoveImageSticker_CV() : #얼굴에 스티커 붙이기
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList, result
    scaler = 0.3

    # 얼굴, 모양 예측
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 비디오 불러오기
    #movieName = askopenfilename(parent=window,
    #                            filetypes=(('동영상 파일', '*.mp4;*.avi'), ('All File', '*.*')))
    #cap = cv2.VideoCapture(movieName)
    cap = cv2.VideoCapture('C:/images/Movies/asian_girl.mp4')
    # 얼굴에 붙일 이미지 (스티커) 불러오기
    sticker = askopenfilename(parent=window,
                              filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))
    overlay = cv2.imread(sticker, cv2.IMREAD_UNCHANGED)

    # 오버레이 기능
    def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
        global cvInImage, cvOutImage, fileList, result

        bg_img = background_img.copy()
        # 3채널을 4채널로 변환하기
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

        # 4채널을 3채널로 변환하기
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img

    face_roi = []
    face_sizes = []

    # loop
    while True:
        # read frame buffer from video
        ret, img = cap.read()
        if not ret:
            break

        # frame 리사이즈
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        # 얼굴 찾기
        if len(face_roi) == 0:
            faces = detector(img, 1)
        else:
            roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
            # cv2.imshow('roi', roi_img)
            faces = detector(roi_img)

        # 얼굴을 인식하지 못했을 때
        if len(faces) == 0:
            print('no faces!')

        # 얼굴 랜드마크 찾기
        for face in faces:
            if len(face_roi) == 0:
                dlib_shape = predictor(img, face)
                shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
            else:
                dlib_shape = predictor(roi_img, face)
                shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

            for s in shape_2d:
                cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # 얼굴 중앙 찾기
            center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

            # 얼굴 경계점 찾기
            min_coords = np.min(shape_2d, axis=0)
            max_coords = np.max(shape_2d, axis=0)

            # draw min, max coords//
            cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            # 얼굴 사이즈
            face_size = max(max_coords - min_coords)
            face_sizes.append(face_size)
            if len(face_sizes) > 10:
                del face_sizes[0]
            mean_face_size = int(np.mean(face_sizes) * 1.8)

            # compute face roi
            face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2),
                                 int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
            face_roi = np.clip(face_roi, 0, 10000)

            # 얼굴 위에 스티커 붙이기
            result = overlay_transparent(ori, overlay, center_x, center_y, overlay_size=(mean_face_size, mean_face_size))

        # visualize
        cv2.imshow('original', ori) # 원본 영상
        cv2.imshow('facial landmarks', img) # 얼굴 랜드마크 점 영상
        cv2.imshow('result', result) # 결과 영상 (얼굴에 스티커 붙이기)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 프로그램 종료하고 창 닫기
    cap.release()
    cv2.destroyAllWindows()

def deepMoveImageEmotion_CV() : # 감정 분석하기
    # Face detection XML load and trained model loading
    face_detection = cv2.CascadeClassifier('harrcascade/haarcascade_frontalface_alt.xml')
    emotion_classifier = load_model('emotion_model.hdf5', compile=False)
    EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

    # 웹캠 사용해서 영상불러오기
    # movieName = cv2.imread('C:/images/Movies/asian_girl.mp4')
    camera = cv2.VideoCapture(0)
    # 영상 출력 화면 크기
    camera.set(3, 640)
    camera.set(4, 480)

    # cv2.VideoCapture(0)

    while True:
        # camera에서 영상 가져오기
        ret, frame = camera.read()

        # grayscale로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 인식하기
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 빈 화면 만들기
        canvas = np.zeros((250, 300, 3), dtype="uint8")

        # 얼굴을 인식했을때만 감정 분석하기
        if len(faces) > 0:
            # For the largest image
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            # neural network를 위해 48x48 사이즈로 이미지 조절
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 감정 인식하기
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # 라벨링
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            # 라벨 출력하기
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # 창 두개 열기
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
        cv2.imshow('Emotion Recognition', frame)
        cv2.imshow("Probabilities", canvas)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 프로그램 종료하고 창 닫기
    camera.release()
    cv2.destroyAllWindows()

def deepMoveImageRemoveBG_CV() : # 배경제거
    # 웹캠으로 동영상 불러오기
    cap = cv2.VideoCapture(0)
    frame_width, frame_height, frame_rate = int(cap.get(3)), int(cap.get(4)), int(cap.get(10))
    # 코덱 설정 / 파일 저장
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('C:/images/Movies/asian_girl-mog2.mp4', fourcc, frame_rate, (frame_width, frame_height), 0)

    # BackgroundSubtractorMOG2 이용해서 배경 제거하기
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while (cap.isOpened()):
        # ret: frame capture 결과(boolean)
        # frame: capture한 frame
        ret, frame = cap.read()
        if (ret):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask[fgmask < 128] = 0
            fgmask[fgmask > 127] = 255
            cv2.imshow('mog2', fgmask)
            out.write(fgmask)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

## 전역 변수부
window, status, canvas, paper = None, None, None, None
inImage, outImage = [], []
inH, inW, outH, outW = [0] * 4
cvInImage, cvOutImage = None, None
filename = ""
RGB ,R, G, B =3, 0, 1, 2
## MySQL DB 관련
cur, conn = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None

## 메인코드부
def main(login) :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    global status

    window = Tk()
    window.title("칼라 영상처리 ver1.5")
    window.geometry("512x512")
    # window.resizable(width=False, height=False)
    status = Label(window, text="이미지정보", bd=1, relief=SUNKEN, anchor=W)
    status.pack(side=BOTTOM, fill=X)

    text1 = Label(window, text="로그인되었습니다")
    text1.pack()

    # 메뉴 만들기
    mainMenu = Menu(window)
    window.configure(menu=mainMenu)

    fileMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="파일", menu=fileMenu)
    fileMenu.add_command(label="열기(open)", command=openFile)
    fileMenu.add_command(label="저장(Save)", command=saveImage)
    fileMenu.add_command(label="실행취소", command=undoImage)
    fileMenu.add_command(label="재실행", command=undoImage)
    fileMenu.add_separator()
    fileMenu.add_command(label="닫기(close)")

    # 하위 메뉴 #
    # 머신러닝 처리 #
    harrMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="머신러닝(Harrcascade)", menu=harrMenu)
    harrMenu.add_command(label="얼굴 인식", command=faceDetect_CV)
    harrMenu.add_command(label="코 인식", command=noseDetect_CV)
    harrMenu.add_command(label="귀 인식", command=earDetect_CV)
    harrMenu.add_command(label="고양이 얼굴 인식", command=catFaceDetect_CV)
    harrMenu.add_command(label="하르케스케이드 얼굴 인식해서 모자이크", command=faceDetectMosaic_CV)

    # 딥러닝 처리 #
    deepCVMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="딥러닝", menu=deepCVMenu)
    deepCVMenu.add_command(label="사물 인식(정지영상)", command=deepStopImage_CV)
    deepCVMenu.add_command(label="사물 인식(동영상)", command=deepMoveImage_CV)
    deepCVMenu.add_command(label="얼굴에 스티커붙이기(동영상)", command=deepMoveImageSticker_CV)
    deepCVMenu.add_command(label="얼굴 감정 분석(동영상)", command=deepMoveImageEmotion_CV)
    deepCVMenu.add_command(label="배경 제거(동영상)", command=deepMoveImageRemoveBG_CV)

    window.mainloop()