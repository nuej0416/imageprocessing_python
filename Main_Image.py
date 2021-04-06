## 이미지 처리하기 ##

from tkinter import *
from tkinter.filedialog import *
from tkinter import filedialog
from tkinter.simpledialog import *
import math
import struct
import cv2
import numpy
import pymysql
import random
import tempfile
import os
import numpy as np
import time
import xlrd
import xlwt
import xlsxwriter

## 함수 선언부
# 공통 함수
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

#MySQL 관련 함수

def upMySQL() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    global fileList, cur, conn

    if filename == None or filename == "":
        return

    saveCvPhoto = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH):
        for k in range(outW):
            tup = tuple(([outImage[B][i][k], outImage[G][i][k], outImage[R][i][k]]))
            saveCvPhoto[i, k] = tup

    saveFname = tempfile.gettempdir() + '/' + os.path.basename(filename)
    cv2.imwrite(saveFname, saveCvPhoto)

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    p_id = random.randint(-2100000000, 2100000000)
    tmpName = os.path.basename(os.path.basename(saveFname))
    p_fname, p_ext = tmpName.split('.')
    p_size = os.path.getsize(saveFname)
    tmpImage = cv2.imread(saveFname)
    p_height = tmpImage.shape[0]
    p_width = tmpImage.shape[1]
    from datetime import datetime
    p_upDate = datetime.today().strftime("%Y%m%d%H%M%S") #구글링 통해서 현재시간 받는법 알아내기
    p_upUser = 'root' # 로그인한 사용자

    # 파일을 읽기

    fp = open(filename, 'rb')
    blobData = fp.read()
    fp.close()

    # 파일 정보 입력

    sql = "INSERT INTO photo_table(p_id, p_fname, p_ext, p_size, p_height, p_width, "
    sql += "p_upDate, p_UpUser, p_photo) VALUES (" + str(p_id) + ", '" + p_fname + "', '" + p_ext
    sql += "', " + str(p_size) + "," + str(p_height) + "," + str(p_width) + ", '" + p_upDate
    sql += "', '" + p_upUser + "', %s )"

    tupleData = (blobData)
    cur.execute(sql, tupleData)

    conn.commit()
    cur.close()
    conn.close()

    messagebox.showinfo('성공', filename + ' 잘 입력됨.')

def upFolderMySQL() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList, cur, conn

    # 목록 준비
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    # 폴더 선택

    folder_dir = filedialog.askdirectory()
    file_list = os.listdir(folder_dir)

    for i in range(len(file_list)) :
        p_id = random.randint(-2100000000, 2100000000)
        filename = folder_dir + '/' + file_list[i]
        tmpName = os.path.basename(filename)
        p_fname, p_ext = tmpName.split('.')
        p_size = os.path.getsize(filename)
        tmpImage = cv2.imread(filename)
        p_height = tmpImage.shape[0]
        p_width = tmpImage.shape[1]
        from datetime import datetime
        p_upDate = datetime.today().strftime("%Y%m%d%H%M%S")  # 구글링 통해서 현재시간 받는법 알아내기
        p_upUser = 'root'  # 로그인한 사용자

        # 파일을 읽기
        fp = open(filename, 'rb')
        blobData = fp.read()
        fp.close()

        # 파일 정보 입력

        sql = "INSERT INTO photo_table(p_id, p_fname, p_ext, p_size, p_height, p_width, "
        sql += "p_upDate, p_UpUser, p_photo) VALUES (" + str(p_id) + ", '" + p_fname + "', '" + p_ext
        sql += "', " + str(p_size) + "," + str(p_height) + "," + str(p_width) + ", '" + p_upDate
        sql += "', '" + p_upUser + "', %s )"

        tupleData = (blobData)
        cur.execute(sql, tupleData)

    conn.commit()
    cur.close()
    conn.close()

    messagebox.showinfo('성공', filename + ' 잘 입력됨.')

def downMySQL() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage,fileList

    # 목록 준비
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    sql = "SELECT p_id, p_fname, p_ext, p_size FROM photo_table"
    cur.execute(sql)

    fileList = cur.fetchall()  # 목록 전부 가져옴

    cur.close()
    conn.close()

    # 서브 윈도우창 나오기
    def downLoad() :
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
        global cvInImage, cvOutImage, fileList
        global fileList, filename
        selectIndex = listData.curselection()[0]  # 튜플의 데이터중 []에 해당하는 요소 선택

        conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
        cur = conn.cursor()

        sql = "SELECT p_fname, p_ext, p_photo FROM photo_table WHERE p_id = "
        sql += str(fileList[selectIndex][0])
        cur.execute(sql)
        p_fname, p_ext, p_photo = cur.fetchone()

        fullPath = tempfile.gettempdir() + '/' + p_fname + '.' + p_ext  # C드라이브의 임시파일폴더에 저장
        fp = open(fullPath, 'wb')
        fp.write(p_photo)
        print(fullPath)
        fp.close()

        cur.close()
        conn.close()
        filename = fullPath
        subWindow.destroy()

        ##
        cvInImage = cv2.imread(filename)
        print(cvInImage)
        inH = cvInImage.shape[0]
        inW = cvInImage.shape[1]

        # 입력이미지용 메모리 할당
        inImage = []
        for _ in range(RGB):
            inImage.append(malloc(inH, inW))

        # 파일 --> 메모리 로딩
        for i in range(inH):
            for k in range(inW):
                inImage[R][i][k] = cvInImage.item(i, k, B)  # opencv는 RGB를 BGR로 표현
                inImage[G][i][k] = cvInImage.item(i, k, G)
                inImage[B][i][k] = cvInImage.item(i, k, R)
        equalColor()
        ##

    subWindow = Toplevel(window)
    subWindow.geometry("300x500")

    ## 스크롤바 나타내기
    frame = Frame(subWindow)
    scrollbar = Scrollbar(frame)
    scrollbar.pack(side='right', fill='y')
    listData = Listbox(frame, yscrollcommand=scrollbar.set);
    listData.pack()
    scrollbar['command'] = listData.yview
    frame.pack()

    for fileTup in fileList:
        listData.insert(END, fileTup[1:])  # 목록에 튜플의 1번째부터 넣어서 보여준다. 여기에서는 p_id제외하고 보여줌
    btnDownLoad = Button(subWindow, text="!!다운로드!!", command=downLoad)
    btnDownLoad.pack(side=LEFT, padx=10, pady=10)

    return

    # 파일 선택하기
    #filename = askopenfilename(parent=window, filetypes=(("Color 파일", "*.jpg;*.png;*.bmp;*.tif;"), ("ALL File", "*.*")))

    ##(중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)

    inH = cvInImage.shape[0]
    inW = cvInImage.shape[1]

    # 입력이미지용 메모리 할당
    inImage = []
    for _ in range(RGB):
        inImage.append(malloc(inH, inW))

    # 파일 --> 메모리 로딩
    for i in range(inH):
        for k in range(inW):
            inImage[R][i][k] = cvInImage.item(i, k, B)  # opencv는 RGB를 BGR로 표현
            inImage[G][i][k] = cvInImage.item(i, k, G)
            inImage[B][i][k] = cvInImage.item(i, k, R)
    equalColor()

### Excel 처리 부분 ###

def saveExcel() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='xls',
                           filetypes=(("엑셀 파일", "*.xls"), ("모든 파일", "*.*")))
    if saveFp == '' or saveFp == None:
        return
    xlsName = saveFp.name
    #sheetName = os.path.basename(filename) #cat01_256.png 이대로 시트이름 저장
    wb = xlwt.Workbook()
    ws_R = wb.add_sheet("RED")
    ws_G = wb.add_sheet("GREEN")
    ws_B = wb.add_sheet("BLUE")

    for i in range(outH):
        for k in range(outW):
            ws_R.write(i, k, outImage[R][i][k])
            ws_G.write(i, k, outImage[G][i][k])
            ws_B.write(i, k, outImage[B][i][k])
    wb.save(xlsName)
    print('Excel. save ok...')

def saveExcel2() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='xls',
                           filetypes=(("엑셀 파일", "*.xls"), ("모든 파일", "*.*")))
    if saveFp == '' or saveFp == None:
        return
    xlsName = saveFp.name
    #sheetName = os.path.basename(filename) #cat01_256.png 이대로 시트이름 저장
    wb = xlsxwriter.Workbook(xlsName)
    ws_R = wb.add_worksheet("RED")
    ws_G = wb.add_worksheet("GREEN")
    ws_B = wb.add_worksheet("BLUE")

    for i in range(outH):
        for k in range(outW):
            ws_R.write(i, k, outImage[R][i][k])
            ws_G.write(i, k, outImage[G][i][k])
            ws_B.write(i, k, outImage[B][i][k])
    #wb.save(xlsName)
    wb.close()
    print('Excel. save ok...')

def openExcel() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    filename = askopenfilename(parent=window,
                               filetypes=(('엑셀 파일', '*.xls'), ('All File', '*.*')))

    workbook = xlrd.open_workbook(filename)
    wsList = workbook.sheets() #3장 워크시트 리스트
    inH = wsList[0].nrows
    inW = wsList[0].ncols

    ## 입력이미지용 메모리 할당
    inImage = []
    for _ in range(RGB):
        inImage.append(malloc(inH, inW))
    ## 파일 --> 메모리 로딩

    for i in range(inH):
        for k in range(inW):
            inImage[R][i][k] = int(wsList[R].cell_value(i, k))
            inImage[G][i][k] = int(wsList[G].cell_value(i, k))
            inImage[B][i][k] = int(wsList[B].cell_value(i, k))

    equalColor()

def drawExcel() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList
    if filename == '' or filename == None:
        return

    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='xls',
                           filetypes=(("엑셀 파일", "*.xls"), ("모든 파일", "*.*")))
    if saveFp == '' or saveFp == None:
        return
    xlsName = saveFp.name
    # sheetName = os.path.basename(filename) #cat01_256.png 이대로 시트이름 저장
    wb = xlsxwriter.Workbook(xlsName)
    ws_R = wb.add_worksheet("RED")
    ws_G = wb.add_worksheet("GREEN")
    ws_B = wb.add_worksheet("BLUE")
    ws_C = wb.add_worksheet("COLOR")

    # 셀 크기 조절하기
    ws_R.set_column(0, outW-1, 1.0) #엑셀에서 사이즈 0.34정도임
    for i in range(outH) :
        ws_R.set_row(i, 9.5) #엑셀에서 약 0.35사이즈 정도

    ws_G.set_column(0, outW - 1, 1.0)  # 엑셀에서 사이즈 0.34정도임
    for i in range(outH):
        ws_G.set_row(i, 9.5)  # 엑셀에서 약 0.35사이즈 정도

    ws_B.set_column(0, outW - 1, 1.0)  # 엑셀에서 사이즈 0.34정도임
    for i in range(outH):
        ws_B.set_row(i, 9.5)  # 엑셀에서 약 0.35사이즈 정도

    ws_C.set_column(0, outW - 1, 1.0)  # 엑셀에서 사이즈 0.34정도임
    for i in range(outH):
        ws_C.set_row(i, 9.5)  # 엑셀에서 약 0.35사이즈 정도

    #메모리 -> 엑셀 파일
    for i in range(outH):
        for k in range(outW):
            # Red 시트
            data = outImage[R][i][k]
            if data <= 15 :
                hexStr = '#' + ('0' +hex(data)[2:]) + '0000'
            else :
                hexStr = '#' + hex(data)[2:] + '0000'
            # 셀 속성 변경
            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_R.write(i,k,'',cell_format)

            # Green 시트
            data = outImage[G][i][k]
            if data <= 15:
                hexStr = '#' + '00' + ('0' +hex(data)[2:]) + '00'
            else:
                hexStr = '#' + '00' + hex(data)[2:] + '00'
            # 셀 속성 변경
            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_G.write(i, k, '', cell_format)

            # Blue 시트
            data = outImage[B][i][k]
            if data <= 15:
                hexStr = '#' + '0000' + ('0' + hex(data)[2:])
            else:
                hexStr = '#' + '0000'+ hex(data)[2:]
            # 셀 속성 변경
            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_B.write(i, k, '', cell_format)

    for i in range(outH):
        for k in range(outW):
            # Red 시트
            data1 = outImage[R][i][k]
            if data1 <= 15:
                hexStr1 = '#' + ('0' + hex(data1)[2:])
            else:
                hexStr1 = '#' + hex(data1)[2:]

            # Green 시트
            data2 = outImage[G][i][k]
            if data2 <= 15:
                hexStr2 = ('0' + hex(data2)[2:])
            else:
                hexStr2 = hex(data2)[2:]

            # Blue 시트
            data3 = outImage[B][i][k]
            if data3 <= 15:
                hexStr3 = ('0' + hex(data3)[2:])
            else:
                hexStr3 = hex(data3)[2:]
            # 셀 속성 변경
            hexStr = hexStr1 + hexStr2 + hexStr3

            cell_format = wb.add_format()
            cell_format.set_bg_color(hexStr)
            ws_C.write(i, k, '', cell_format)

    wb.close()
    print('Excel Art. save ok...')

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

def addColor() : #밝게하기 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    value = askinteger("밝게 할 값 : ", "값 ->", minvalue=1, maxvalue=255)
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                out = inImage[rgb][i][k] + value
                if out > 255:
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = out
    displayImageColor()

def subColor() : #어둡게하기 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename

    # 파일 열지 않고 다른 기능 실행할때 아무 반응없이 넘어가기
    if filename == "" or filename == None:
        return
    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    value = askinteger("어둡게 할 값 : ", "값 ->", minvalue=1, maxvalue=255)
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                out = inImage[rgb][i][k] - value
                if out < 0 :
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = out
    displayImageColor()

def grayColor() : #color를 grayscale로 바꾸는 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for i in range(inH) :
        for k in range(inW) :
            c = inImage[R][i][k] + inImage[G][i][k] + inImage[B][i][k]
            c = int(c/3)
            outImage[R][i][k] = outImage[G][i][k] = outImage[B][i][k] = c
    displayImageColor()

def gammaColor() : #감마연산 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # out  = in ** (1/r)
    r = askfloat("감마연산 : ", "값 ->")

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                v = inImage[rgb][i][k] ** (1 / r)
                if v > 255:
                    outImage[rgb][i][k] = 255
                elif v < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(v)
    displayImageColor()

def paraCapColor() : #파라볼라 캡 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # outImage = 255.0 * (( inImage/128.0 - 1.0)**2)

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                v = 255.0 * ((inImage[rgb][i][k] / 128.0 - 1.0) ** 2)
                if v > 255:
                    outImage[rgb][i][k] = 255
                elif v < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(v)
    displayImageColor()

def paraCupColor() : #파라볼라 컵 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # outImage = 255.0 - (255.0 * (( inImage/128.0 - 1.0)**2))

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                v = 255.0 - (255.0 * ((inImage[rgb][i][k] / 128.0 - 1.0) ** 2))
                if v > 255:
                    outImage[rgb][i][k] = 255
                elif v < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(v)
    displayImageColor()

def binColor() : #이진화(127기준) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                if inImage[rgb][i][k] > 127:
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = 0
    displayImageColor()

def avrColor() : #이진화(평균값기준) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for i in range(inH) :
        for k in range(inW) :
            c = inImage[R][i][k] + inImage[G][i][k] + inImage[B][i][k]
            c = int(c/3)
            outImage[R][i][k] = outImage[G][i][k] = outImage[B][i][k] = c

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                if inImage[rgb][i][k] > c:
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = 0
    displayImageColor()

def medColor() : #이진화(중위수기준) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    mid = 0
    tmpAry = []
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                tmpAry.append(inImage[rgb][i][k])
    tmpAry.sort()
    mid = tmpAry[int((inH * inW) / 2)]

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                if inImage[rgb][i][k] > mid:
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = 0
    displayImageColor()

def point2Color() : #범위강조 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    p1 = askinteger("", "값 -->")
    p2 = askinteger("", "값 -->")
    # 변수 서로 바꾸는 식 (파이썬)
    if p1 > p2:
        p1, p2 = p2, p1

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                if p1 < inImage[rgb][i][k] < p2 :
                    outImage[rgb][i][k] = 255
                else :
                    outImage[rgb][i][k] = inImage[rgb][i][k]
    displayImageColor()

def revColor() : #영상반전 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][i][k] =255 - inImage[rgb][i][k]
    displayImageColor()

#####기하학 처리#####
def moveColor() : #영상이동 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    dy = askinteger("", "x 변위 : ")
    dx = askinteger("", "y 변위 : ")
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                if 0 <= i + dx < outH and 0 <= k + dy < outW :
                    outImage[rgb][i+dx][k+dy] = inImage[rgb][i][k]
    displayImageColor()

def zoomOutColor() : #영상 축소(정방형 이미지) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    scale = askinteger("축소", "배율 : ") #짝수만 처리 가능

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = int(inH / scale)
    outW = int(inW / scale)

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][int(i / scale)][int(k / scale)] = inImage[rgb][i][k]

    displayImageColor()

def zoomOut2Color() : #영상 축소(백워딩. 정방형 장방형 둘다 가능) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    scale = askinteger("축소", "배율 : ") #짝수만 처리 가능

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = int(inH / scale)
    outW = int(inW / scale)

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(outH) :
            for k in range(outW) :
                outImage[rgb][i][k] = inImage[rgb][i * scale][k * scale]
    displayImageColor()

def zoomInColor() : #영상 확대 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    scale = askinteger("확대", "배율 : ") #짝수만 처리 가능

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = int(inH * scale)
    outW = int(inW * scale)

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][i * scale][k * scale] = inImage[rgb][i][k]
    displayImageColor()

def zoomIn2Color() : #영상 확대(백워딩. 정방형 장방형 둘다 가능) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    scale = askinteger("확대", "배율 : ") #짝수만 처리 가능

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = int(inH * scale)
    outW = int(inW * scale)

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(outH) :
            for k in range(outW) :
                outImage[rgb][i][k] = inImage[rgb][int(i / scale)][int(k / scale)]
    displayImageColor()

def mirLRColor() : #미러링(좌우) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    tmp = []
    for i in range(inH) :
        tmp = []
        for k in range(inW) :
            tmp.append(0)
        outImage.append(tmp)
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][i][k] = inImage[rgb][i][inW - k -1]
    displayImageColor()

def mirTBColor() : #미러링(상하) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    tmp = []
    for i in range(inH) :
        tmp = []
        for k in range(inW) :
            tmp.append(0)
        outImage.append(tmp)
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][i][k] = inImage[rgb][inH - i - 1][k]
    displayImageColor()

def rotateColor() : #영상 회전 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
        # 회전 공식 #
    # xd = cos * xs - sin * ys
    # yd = sin * xs + sin * ys
    angle = askinteger("회전", "각도 --> ", minvalue=0, maxvalue=360)
    r = angle * math.pi / 180  # 라디안 값 구하기

    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                xs = i
                ys = k
                xd = int(math.cos(r) * xs - math.sin(r) * ys)
                yd = int(math.sin(r) * xs + math.cos(r) * ys)
                if 0 <= xd < outH and 0 <= yd < outW:
                    outImage[rgb][xd][yd] = inImage[rgb][xs][ys]  # Out = In
    displayImageColor()

def rotate2Color():  #영상 회전(중심, 백워딩) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # 회전 공식 #
    # xd = cos * xs - sin * ys
    # yd = sin * xs + sin * ys
    angle = askinteger("회전", "각도 --> ", minvalue=0, maxvalue=360)
    r = angle * math.pi / 180  # 라디안 값 구하기
    # 중심 찾기
    cx = inH // 2
    cy = inW // 2

    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                xs = i
                ys = k
                xd = int(math.cos(r) * (xs - cx) - math.sin(r) * (ys - cy) + cx)  # o,o 으로 중심점을 이동했다가 다시 결과를 중심점으로 이동
                yd = int(math.sin(r) * (xs - cx) + math.cos(r) * (ys - cy) + cy)
                if 0 <= xd < outH and 0 <= yd < outW:
                    outImage[rgb][xs][ys] = inImage[rgb][xd][yd]  # Out = In
    displayImageColor()

#####화소영역 처리#####
def embossColor() : #엠보싱 효과 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # (중요!) 마스크
    msize = 3
    mask = [[-1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]]
    tmpInImage, tmpOutImage = [], []
    for _ in range(RGB) :
        tmpInImage.append(malloc(inH + 2, inW + 2, 127))
    for _ in range(RGB):
        tmpOutImage.append(malloc(outH, outW))

    # inInimage --> tmpInImage
    for rgb in range(RGB) :
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(msize):
                    for n in range(msize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # 마무리 --> 마스크에 따라서 127 더할지 결정
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127.0
        # tmpOutImage --> outImage : overflow 처리
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    displayImageColor()

def blurrColor() : #블러 효과 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # (중요!) 마스크
    msize = 3
    mask = [[1 / 9.0, 1 / 9.0, 1 / 9.0],
            [1 / 9.0, 1 / 9.0, 1 / 9.0],
            [1 / 9.0, 1 / 9.0, 1 / 9.0]]
    tmpInImage, tmpOutImage = [], []
    for _ in range(RGB):
        tmpInImage.append(malloc(inH + 2, inW + 2, 127))
    for _ in range(RGB):
        tmpOutImage.append(malloc(outH, outW))

    # inInimage --> tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(msize):
                    for n in range(msize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # # 마무리 --> 마스크에 따라서 127 더할지 결정
    # for rgb in range(RGB):
    #     for i in range(outH):
    #         for k in range(outW):
    #             tmpOutImage[rgb][i][k] += 127.0
        # tmpOutImage --> outImage : overflow 처리
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    displayImageColor()

def gaussianColor() : #가우시안필터 효과알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # (중요!) 마스크
    msize = 3
    mask = [[1 / 16.0, 1 / 8.0, 1 / 16.0],
            [1 / 8.0, 1 / 4.0, 1 / 8.0],
            [1 / 16.0, 1 / 8.0, 1 / 16.0]]
    tmpInImage, tmpOutImage = [], []
    for _ in range(RGB):
        tmpInImage.append(malloc(inH + 2, inW + 2, 127))
    for _ in range(RGB):
        tmpOutImage.append(malloc(outH, outW))

    # inInimage --> tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(msize):
                    for n in range(msize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # # 마무리 --> 마스크에 따라서 127 더할지 결정
    # for rgb in range(RGB):
    #     for i in range(outH):
    #         for k in range(outW):
    #             tmpOutImage[rgb][i][k] += 127.0
        # tmpOutImage --> outImage : overflow 처리
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    displayImageColor()

def sharpColor() : #샤프닝 효과 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # (중요!) 마스크
    tmask = askinteger("샤프닝 강조", "1(강하게) or 2(약하게) 입력 : ")
    msize = 3
    mask1 = [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]]
    mask2 = [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]]
    if tmask == 1:
        mask = mask1
    elif tmask == 2:
        mask = mask2
    tmpInImage, tmpOutImage = [], []
    for _ in range(RGB):
        tmpInImage.append(malloc(inH + 2, inW + 2, 127))
    for _ in range(RGB):
        tmpOutImage.append(malloc(outH, outW))

    # inInimage --> tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(msize):
                    for n in range(msize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # # 마무리 --> 마스크에 따라서 127 더할지 결정
    # for rgb in range(RGB):
    #     for i in range(outH):
    #         for k in range(outW):
    #             tmpOutImage[rgb][i][k] += 127.0
        # tmpOutImage --> outImage : overflow 처리
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    displayImageColor()

def sharp2Color() : #샤프닝 (고주파 통과 필터) 효과 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # (중요!) 마스크
    msize = 3
    mask = [[-1 / 9.0, -1 / 9.0, -1 / 9.0],
            [-1 / 9.0, 8 / 9.0, -1 / 9.0],
            [-1 / 9.0, -1 / 9.0, -1 / 9.0]]
    tmpInImage, tmpOutImage = [], []
    for _ in range(RGB):
        tmpInImage.append(malloc(inH + 2, inW + 2, 127))
    for _ in range(RGB):
        tmpOutImage.append(malloc(outH, outW))

    # inInimage --> tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(msize):
                    for n in range(msize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # # 마무리 --> 마스크에 따라서 127 더할지 결정
    # for rgb in range(RGB):
    #     for i in range(outH):
    #         for k in range(outW):
    #             tmpOutImage[rgb][i][k] += 127.0
        # tmpOutImage --> outImage : overflow 처리
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    displayImageColor()

def sharp3Color() : #샤프닝 (저주파 통과 필터) 효과 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # (중요!) 마스크
    msize = 3
    mask = [[1 / 9.0, 1 / 9.0, 1 / 9.0],
            [1 / 9.0, 1 / 9.0, 1 / 9.0],
            [1 / 9.0, 1 / 9.0, 1 / 9.0]]
    tmpInImage, tmpOutImage = [], []
    for _ in range(RGB):
        tmpInImage.append(malloc(inH + 2, inW + 2, 127))
    for _ in range(RGB):
        tmpOutImage.append(malloc(outH, outW))

    # inInimage --> tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(msize):
                    for n in range(msize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # # 마무리 --> 마스크에 따라서 127 더할지 결정
    # for rgb in range(RGB):
    #     for i in range(outH):
    #         for k in range(outW):
    #             tmpOutImage[rgb][i][k] += 127.0
        # tmpOutImage --> outImage : overflow 처리
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    displayImageColor()

#####히스토그램#####
def stretchColor() : #히스토그램 스트레칭 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # 히스토그램 스트레칭 공식
    # out = (in - low) / (high - low) * 255.0
    # low , high 값 구하기
    low = high = inImage[0][0][0]
    for rgb in range(RGB) :
        for i in range(inH):
            for k in range(inW):
                if low > inImage[rgb][i][k] :
                    low = inImage[rgb][i][k]
                elif high < inImage[rgb][i][k] :
                    high = inImage[rgb][i][k]
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                out = (inImage[rgb][i][k] - low) / (high - low) * 255.0
                # overflow 체크
                if out > 255:
                    outImage[rgb][i][k] = 255
                elif out < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(out)  # 실수는 마지막에 정수로 변환
    displayImageColor()

def endInColor() : #엔드인 탐색 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 진짜 영상처리 알고리즘
    # 히스토그램 스트레칭 공식
    # out = (in - low) / (high - low) * 255.0
    # low , high 값 구하기
    low = high = inImage[0][0][0]
    for rgb in range(RGB) :
        for i in range(inH):
            for k in range(inW):
                if low > inImage[rgb][i][k] :
                    low = inImage[rgb][i][k]
                elif high < inImage[rgb][i][k] :
                    high = inImage[rgb][i][k]
    low += 50
    high -= 50
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                out = (inImage[rgb][i][k] - low) / (high - low) * 255.0
                # overflow 체크
                if out > 255:
                    outImage[rgb][i][k] = 255
                elif out < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(out)  # 실수는 마지막에 정수로 변환
    displayImageColor()

def equalizedColor() : # 히스토그램 평활화 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    # 1단계 : 히스토그램 만들기
    histo = [0 for _ in range(256)]
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                histo[inImage[rgb][i][k]] += 1

    # 2단계 : 누적 히스토그램 만들기
    sumhisto = [0 for _ in range(256)]
    sumhisto[0] = histo[0]
    for i in range(1, 256):
        sumhisto[i] = histo[i] + sumhisto[i - 1]

    # 3단계 : 정규화 히스토그램 만들기
    # n = 누적합 * ( 1 / 픽셀수 ) * 최대 value
    # n = 누적합 * ( 1 / (inH+inW)) * 255
    normalHisto = [0 for _ in range(256)]
    for i in range(256):
        normalHisto[i] = sumhisto[i] * (1 / (inH * inW)) * 255.0
    # 진짜 영상처리 알고리즘
    for rgb in range(RGB) :
        for i in range(inH):
            for k in range(inW):
                #outImage[rgb][i][k] = int(normalHisto[inImage[rgb][i][k]])  # Out = In

                # overflow 체크
                if normalHisto[inImage[rgb][i][k]] > 255:
                    outImage[rgb][i][k] = 255
                elif normalHisto[inImage[rgb][i][k]] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(normalHisto[inImage[rgb][i][k]])

    displayImageColor()

#####특수효과#####
def click(): # 스티커 꾸미기 / X,Y 좌표를 받을 함수
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, xPoint, yPoint
    global cvInImage, cvOutImage
    messagebox.showinfo('위치선택','원하는 위치를 클릭하세요')
    canvas.bind("<Button>",sticker)   # 캔버스 화면의 아무데나 누르면 sticker함수 호출
    canvas.mainloop()
    ########################

def sticker(event) : # 스티커 붙이기 (덮어쓰기)
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, xPoint, yPoint
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## 스티커 선택하기
    messagebox.showinfo('스티커 선택','스티커 이미지를 선택하세요')
    # 파일 선택하기
    filename = askopenfilename(parent=window,
           filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))
    # (중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)
    skH = cvInImage.shape[0]
    skW = cvInImage.shape[1]
    ## 스티커이미지용 메모리 할당
    skImage = mallocNumpy(RGB, skH, skW)

    ## 스티커파일 --> 메모리 로딩
    for i in range(skH):
        for k in range(skW):
            skImage[R][i][k] = cvInImage.item(i, k ,B)
            skImage[G][i][k] = cvInImage.item(i, k, G)
            skImage[B][i][k] = cvInImage.item(i, k, R)
    # 흰부분을 일부러 오버플로우 시켜주기
    for m in range(skH):
        for n in range(skW):
            if skImage[0][m][n]==255  and skImage[1][m][n] ==255 and skImage[2][m][n] ==255 :
                    skImage[0][m][n] = 300
                    skImage[1][m][n] = 300
                    skImage[2][m][n] = 300
    # 전역변수에 저장된 이벤트객체의 x,y 좌표 가져오기
    xPoint, yPoint = int(event.y), int(event.x)
    print(str(xPoint),str(yPoint))  # 잘 가져왔는지 콘솔 확인용
    for rgb in range(RGB):
        for m in range(skH):
          for n in range(skW):  #오버플로우 값을 가진 흰 부분을 제외한 나머지를 x,y 좌표부터 덮어쓰기
                 if (n+yPoint)<= outW and m+xPoint <= outH and  skImage[rgb][m][n] <=255 :
                    outImage[rgb][m+xPoint][n+yPoint] = skImage[rgb][m][n]
    print('stikerColor success')   # 확인용 콘솔출력
    displayImageColor()

### numPy 변환 함수 부분 ###

def addColor_NP() : # numPy 밝게하기 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
    value = askinteger("밝게 할 값 : ", "값 ->", minvalue=1, maxvalue=255)
    if value == None :
        return
    inImage = inImage.astype(np.int16)
    outImage = inImage + value

    # 조건으로 범위 지정
    outImage = np.where(outImage> 255, 255, outImage)
    outImage = np.where(outImage<0, 0, outImage)

    displayImageColor()

def subColor_NP() : # numPy 어둡게하기 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
    value = askinteger("밝게 할 값 : ", "값 ->", minvalue=1, maxvalue=255)
    if value == None :
        return
    inImage = inImage.astype(np.int16)
    outImage = inImage - value

    # 조건으로 범위 지정
    outImage = np.where(outImage> 255, 255, outImage)
    outImage = np.where(outImage<0, 0, outImage)

    displayImageColor()

def grayColor_NP() : #color를 grayscale로 바꾸는 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
    inImage[0] = (inImage[0] + inImage[1] + inImage[2]) // 3
    outImage = np.array([inImage[0], inImage[0], inImage[0]])

    displayImageColor()

def gammaColor_NP() : #감마연산 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
    # out  = in ** (1/r)
    r = askfloat("감마연산 : ", "값 ->")

    outImage = inImage ** (1 /r)
    outImage = outImage.astype(np.int16)

    # 조건으로 범위 지정
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)
    displayImageColor()

def paraCapColor_NP() : #파라볼라 캡 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

        ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
        outH = inH
        outW = inW

        # 출력이미지 메모리 할당
        outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
    # outImage = 255.0 * (( inImage/128.0 - 1.0)**2)

    outImage = 255.0 * ((inImage / 128.0 - 1.0) ** 2)
    outImage = outImage.astype(np.int16)

    # 조건으로 범위 지정
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)

    displayImageColor()

def paraCupColor_NP() : #파라볼라 컵 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
    # outImage = 255.0 - (255.0 * (( inImage/128.0 - 1.0)**2))

    outImage = 255.0 * (255.0 * ((inImage / 128.0 - 1.0) ** 2))
    outImage = outImage.astype(np.int16)

    # 조건으로 범위 지정
    outImage = np.where(outImage > 255, 255, outImage)
    outImage = np.where(outImage < 0, 0, outImage)


    displayImageColor()

def binColor_NP() : #이진화(127기준) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
        # 조건으로 범위 지정
    outImage = np.where(inImage > 128, 255, 0)

    displayImageColor()

def avgColor_NP() : #이진화(평균기준) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
        # 조건으로 범위 지정
    outImage = np.where(inImage > np.average(inImage), 255, 0)

    displayImageColor()

def medColor_NP() : #이진화(중위수기준) 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == "" or filename == None:
        return

    ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW

    # 출력이미지 메모리 할당
    outImage = allocateOutMemory()

    # 진짜 영상처리 알고리즘
        # 조건으로 범위 지정
    outImage = np.where(inImage > np.median(inImage), 255, 0)

    displayImageColor()

# def zoomOutColor_NP() : #영상 축소(정방형 이미지) 알고리즘
#     global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
#     global cvInImage, cvOutImage
#
#     if filename == "" or filename == None:
#         return
#
#     scale = askinteger("축소", "배율 : ") #짝수만 처리 가능
#
#     ##(중요!)출력이미지의 높이, 폭을 결정 --> 알고리즘에 의존
#     outH = int(inH / scale)
#     outW = int(inW / scale)
#
#     # 출력이미지 메모리 할당
#     outImage = allocateOutMemory()
#
#     # 진짜 영상처리 알고리즘
#     pass
#     #outImage = outImage.astype(np.int16)
#
#     # 조건으로 범위 지정
#     outImage = np.where(outImage > 255, 255, outImage)
#     outImage = np.where(outImage < 0, 0, outImage)
#
#     displayImageColor()

def revColor_NP() : # numPy 이미지 반전 알고리즘
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage

    if filename == '' or filename == None:
        return
    start = time.time()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당

    ### 진짜 영상처리 알고리즘 ###

    outImage = 255 - inImage

    ########################
    displayImageColor()
    # 시간재기
    end = time.time()
    second = end - start
    status.configure(text="{0:.2f}".format(second)  + "초  " + status.cget("text"))

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

def grayscale_CV() : # openCV이용 Grayscale
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    cvOut2outImage()
    ########################
    displayImageColor()

def cartoon_CV() : # openCV이용 카툰이미지 효과
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2GRAY)
    cvOutImage = cv2.medianBlur(cvOutImage, 7)
    edges = cv2.Laplacian(cvOutImage, cv2.CV_8U, ksize = 5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    cvOutImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cvOut2outImage()
    ########################
    displayImageColor()

def cartoon1_CV() : # openCV이용 카툰이미지 컬러 효과
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.stylization(cvInImage, sigma_s=100, sigma_r=0.9)

    cvOut2outImage()
    ########################
    displayImageColor()

def gaussian_CV() : # openCV이용 가우시안블러 효과
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.GaussianBlur(cvInImage, (7,7), 0)

    cvOut2outImage()
    ########################
    displayImageColor()

def median_CV() : # openCV이용 메디안 블러 효과
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.medianBlur(cvInImage, 5)

    cvOut2outImage()
    ########################
    displayImageColor()

def detectEdges_CV() : # openCV이용 엣지검출 효과
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.Canny(cvInImage, 100, 200)
    cvOut2outImage()
    ########################
    displayImageColor()

def mirrorRl_CV() : # openCV이용 이미지 좌우 미러링
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.flip(cvInImage, 2)
    cvOut2outImage()
    ########################
    displayImageColor()

def mirrorUd_CV() : # openCV이용 이미지 상하 미러링
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.flip(cvInImage, 0)
    cvOut2outImage()
    ########################
    displayImageColor()

def reflect_CV() : # openCV이용 이미지 반전
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.cvtColor(cvInImage, cv2.COLOR_BGR2RGB)
    cvOut2outImage()
    ########################
    displayImageColor()

def rotate90_CV() : # openCV이용 이미지 90회전
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.rotate(cvInImage, cv2.ROTATE_90_CLOCKWISE) #시계방향으로 90도 회전
    cvOut2outImage()
    ########################
    displayImageColor()

def rotate180_CV() : # openCV이용 이미지 180회전
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.rotate(cvInImage, cv2.ROTATE_180) #180도 회전
    cvOut2outImage()
    ########################
    displayImageColor()

def rotate270_CV() : # openCV이용 이미지 270회전
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    cvOutImage = cv2.rotate(cvInImage, cv2.ROTATE_90_COUNTERCLOCKWISE) #반시계방향으로 90도 회전
    cvOut2outImage()
    ########################
    displayImageColor()

def rotate_CV() : # openCV이용 이미지 회전
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, fileList

    if filename == '' or filename == None:
        return
    ## openCV용 영상처리 ##
    value = askinteger("이미지 회전", "회전시킬 각도")
    if value == None:
        return
    h, w, c = cvInImage.shape
    cvOutImage = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), value, 1)
    cvOutImage = cv2.warpAffine(cvInImage, cvOutImage, (h, w))
    cvOut2outImage()
    ########################
    displayImageColor()

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

    # equalwin = Label(window)
    # equalwin.configure(image = displayImageColor())
    # equalwin.pack(side = LEFT)

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

    MySQLMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="MySQL", menu=MySQLMenu)
    MySQLMenu.add_command(label="MySQL에 파일 저장", command=upMySQL)
    MySQLMenu.add_command(label="MySQL에 폴더 저장", command=upFolderMySQL)
    MySQLMenu.add_command(label="MySQL에서 열기", command=downMySQL)

    excelMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="Excel", menu=excelMenu)
    excelMenu.add_command(label="Excel에 저장", command=saveExcel)
    excelMenu.add_command(label="Excel에 저장(큰파일)", command=saveExcel2)
    excelMenu.add_command(label="Excel에서 열기", command=openExcel)
    excelMenu.add_separator()
    excelMenu.add_command(label="Excel 아트", command=drawExcel)

    ## 상위 메뉴 ##
    imgPCMenu = Menu(mainMenu)
    mainMenu.add_cascade(label = "영상처리", menu=imgPCMenu)
    pixelMenu = Menu(imgPCMenu)
    imgPCMenu.add_cascade(label="화소점 처리", menu=pixelMenu)
    geometryMenu = Menu(imgPCMenu)
    imgPCMenu.add_cascade(label="기하학 처리", menu=geometryMenu)
    areaMenu = Menu(imgPCMenu)
    imgPCMenu.add_cascade(label="화소영역 처리", menu=areaMenu)
    histoMenu = Menu(imgPCMenu)
    imgPCMenu.add_cascade(label="히스토그램", menu=histoMenu)
    specialMenu = Menu(imgPCMenu)
    imgPCMenu.add_cascade(label="특수효과", menu=specialMenu)

    # 하위 메뉴 #
    # 화소점 처리 #
    pixelMenu.add_command(label="동일영상", command=equalColor)
    pixelMenu.add_separator()
    brightMenu = Menu(pixelMenu)
    pixelMenu.add_cascade(label="밝기 조절", menu=brightMenu)
    brightMenu.add_command(label="밝게 하기", command=addColor)
    brightMenu.add_command(label="어둡게 하기", command=subColor)
    pixelMenu.add_separator()
    pixelMenu.add_command(label="그레이스케일", command=grayColor)
    pixelMenu.add_separator()
    pixelMenu.add_command(label="감마 연산", command=gammaColor)
    pixelMenu.add_separator()
    pixelMenu.add_command(label="파라볼라 (Cap)", command=paraCapColor)
    pixelMenu.add_command(label="파라볼라 (Cup)", command=paraCupColor)
    pixelMenu.add_separator()
    binMenu = Menu(pixelMenu)
    pixelMenu.add_cascade(label="이진화 처리", menu=binMenu)
    binMenu.add_command(label="기본 이진화", command=binColor)
    binMenu.add_command(label="평균값 이진화", command=avrColor)
    binMenu.add_command(label="중위수 이진화", command=medColor)
    pixelMenu.add_separator()
    pixelMenu.add_command(label="범위 강조 변환", command=point2Color)
    pixelMenu.add_separator()
    pixelMenu.add_command(label="영상반전", command=revColor)

    # 기하학 처리 #
    geometryMenu.add_command(label="영상 이동", command=moveColor)
    geometryMenu.add_separator()
    sizeMenu = Menu(geometryMenu)
    geometryMenu.add_cascade(label="크기 조절", menu=sizeMenu)
    sizeMenu.add_command(label="영상 축소", command=zoomOutColor)
    sizeMenu.add_command(label="영상 축소 (백워딩)", command=zoomOut2Color)
    sizeMenu.add_separator()
    sizeMenu.add_command(label="영상 확대", command=zoomInColor)
    sizeMenu.add_command(label="영상 확대 (백워딩)", command=zoomIn2Color)
    # geometryMenu.add_command(label="영상 확대 (보간법)_수정중", command=zoomIn3Color)
    geometryMenu.add_separator()
    geometryMenu.add_command(label="영상 미러링(좌우)", command=mirLRColor)
    geometryMenu.add_command(label="영상 미러링(상하)", command=mirTBColor)
    geometryMenu.add_separator()
    geometryMenu.add_command(label="영상 회전", command=rotateColor)
    geometryMenu.add_command(label="영상 회전(중심, 백워딩)", command=rotate2Color)

    # 화소영역 처리 #
    filterMenu = Menu(areaMenu)
    areaMenu.add_cascade(label="필터링", menu=filterMenu)
    filterMenu.add_command(label="엠보싱 효과", command=embossColor)
    filterMenu.add_command(label="블러링 효과", command=blurrColor)
    filterMenu.add_command(label="가우시안필터 효과", command=gaussianColor)
    areaMenu.add_separator()
    sharpMenu = Menu(areaMenu)
    areaMenu.add_cascade(label="샤프닝", menu=sharpMenu)
    sharpMenu.add_command(label="샤프닝 효과", command=sharpColor)
    sharpMenu.add_command(label="고주파 통과 필터", command=sharp2Color)
    sharpMenu.add_command(label="저주파 통과 필터", command=sharp3Color)
    # areaMenu.add_command(label="샤프닝 효과 (Unsharp Masking)_수정중", command=sharp4Color)

    # 히스토 그램 #
    histoMenu.add_command(label="히스토그램 스트레칭", command=stretchColor)
    histoMenu.add_separator()
    histoMenu.add_command(label="앤드 인 탐색", command=endInColor)
    histoMenu.add_separator()
    histoMenu.add_command(label="히스토그램 평활화", command=equalizedColor)

    # 특수효과 #
    specialMenu.add_command(label="스티커 붙이기", command=click)


    # 넘파이 처리 #
    numPyMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="numPy", menu=numPyMenu)
    numPyMenu.add_command(label="밝게 하기", command=addColor_NP)
    numPyMenu.add_command(label="어둡게 하기", command=subColor_NP)
    numPyMenu.add_command(label="그레일스케일", command=grayColor_NP)
    numPyMenu.add_command(label="감마 연산", command=gammaColor_NP)
    numPyMenu.add_command(label="파라볼라 캡", command=paraCapColor_NP)
    numPyMenu.add_command(label="파라볼라 컵", command=paraCupColor_NP)
    numPyMenu.add_separator()
    numPyMenu.add_command(label="기본 이진화", command=binColor_NP)
    numPyMenu.add_command(label="평균 이진화", command=avgColor_NP)
    numPyMenu.add_command(label="중위수 이진화", command=medColor_NP)
    numPyMenu.add_command(label="이미지 반전", command=revColor_NP)

    # openCV 처리 #
    openCVMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="OpenCV", menu=openCVMenu)
    openCVMenu.add_command(label="그레이스케일", command=grayscale_CV)
    openCVMenu.add_command(label="카툰이미지(흑백) 효과", command=cartoon_CV)
    openCVMenu.add_command(label="카툰이미지(컬러) 효과", command=cartoon1_CV)
    openCVMenu.add_separator()
    openCVMenu.add_command(label="가우시안블러 효과", command=gaussian_CV)
    openCVMenu.add_command(label="Median Blur 효과", command=median_CV)
    openCVMenu.add_command(label="엣지검출 효과", command=detectEdges_CV)
    openCVMenu.add_separator()
    openCVMenu.add_command(label="이미지 좌우 미러링", command=mirrorRl_CV)
    openCVMenu.add_command(label="이미지 상하 미러링", command=mirrorUd_CV)
    openCVMenu.add_command(label="이미지 반전", command=reflect_CV)
    openCVMenu.add_separator()
    openCVMenu.add_command(label="이미지 90도 회전", command=rotate90_CV)
    openCVMenu.add_command(label="이미지 180도 회전", command=rotate180_CV)
    openCVMenu.add_command(label="이미지 270도 회전", command=rotate270_CV)
    openCVMenu.add_command(label="이미지 회전", command=rotate_CV)

    window.mainloop()