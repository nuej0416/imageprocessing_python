## 첫화면 ##

import Main_Image
import Main_DL
import Signin
import Root
import pymysql
from tkinter import *
import tkinter.font
from tkinter import messagebox

##함수 선언부
def clickBtnRoot() : # 관리자페이지 로그인
    if edtId.get() == 'admin' and edtPw.get() == '1234' :
        #messagebox.showerror('성공', '로그인 성공')
        print("관리자 페이지 로그인 성공")
        window.destroy()
        Root.main()
    else :
        messagebox.showerror('실패', '로그인 실패')
        print('로그인 실패')
        pass

def login() :
    pass

def clickBtnImage() : # 이미지 처리하기
    global fileList

    # ID와 PW 일치하는지 비교
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    # 아이디, 비밀번호 입력 받기
    loginId = edtId.get()
    loginPw = edtPw.get()

    # DB에서 비교하기
    sql = "SELECT * FROM user_table WHERE u_Id = %s"
    rows_count = cur.execute(sql, loginId)

    if rows_count > 0:
        user_info = cur.fetchone()
        print("user_info : ", user_info)

        if user_info[1] == loginPw:
            #messagebox.showerror('성공', '로그인 성공')
            print("로그인 성공")
            window.destroy()
            Main_Image.main(login)
    else:
        messagebox.showerror('실패', '로그인 실패')
        print("로그인 실패")
        pass
    conn.commit()
    cur.close()
    conn.close()

def clickBtnDL() :
    global fileList

    # ID와 PW 일치하는지 비교
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    # 아이디, 비밀번호 입력 받기
    loginId = edtId.get()
    loginPw = edtPw.get()

    # DB에서 비교하기
    sql = "SELECT * FROM user_table WHERE u_Id = %s"
    rows_count = cur.execute(sql, loginId)

    if rows_count > 0:
        user_info = cur.fetchone()
        print("user_info : ", user_info)

        if user_info[1] == loginPw:
            print("로그인 성공")
            window.destroy()
            Main_DL.main(login)
    else:
        print("로그인 실패")
        pass

    conn.commit()
    cur.close()
    conn.close()

def clickBtnNew() :
    #window.destroy()
    Signin.main()

##전역 변수부
## MySQL DB 관련
cur, conn = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None

##메인 코드부
window = Tk()
window.title("영상처리프로그램")
window.geometry('500x400')
#window.configure(background = "lightblue")

#배경 이미지
wall = PhotoImage(file = "C:/images/wall/wall.gif")
wall_label = Label(image = wall)
wall_label.place(x = -2, y = -2)

# 폰트 설정
font = tkinter.font.Font(family = "맑은 고딕", size = 20)
#제목
titleLabel = Label(window, text = "영상처리프로그램", font = font)
titleLabel.grid(row = 0, column = 2, sticky = W+E+N+S, pady = 10)
titleLabel.configure(background = "#8eceff")

#관리자 버튼
btnRoot = Button(window, text='관리자전용', command=clickBtnRoot)
btnRoot.grid(row = 1, column = 3, sticky = E+N, padx = 5, pady = 5)

#ID 입력 창
tmpId = Label(window)
tmpId.grid(row = 2, column = 0, padx = 20, pady = 20)
tmpId.configure(background = "#8eceff")

textId = Label(window, text = "아이디")
textId.grid(row = 2, column = 1, padx = 5, pady = 3)
textId.configure(background = "#8eceff")

edtId = Entry(window, width = 15)
edtId.grid(row = 2, column = 2, padx = 5, pady = 3)

#PW 입력 창
textPw = Label(window, text = "비밀번호")
textPw.grid(row = 3, column = 1, padx = 5, pady = 3)
textPw.configure(background = "#8eceff")

edtPw = Entry(window, width = 15, show = "*")
edtPw.grid(row = 3, column = 2, padx = 5, pady = 3)

#이미지 처리하기 버튼
btnImage = Button(window, text='이미지 처리하기', command=clickBtnImage)
btnImage.grid(row = 4, column = 2, sticky = W+E+N+S, padx = 5, pady = 7)
# listData = Listbox(window)
# listData.grid(row = 4, column = 3, padx = 5, pady = 7)

#딥러닝 처리하기 버튼
btnDL = Button(window, text='딥러닝 처리하기', command=clickBtnDL)
btnDL.grid(row = 5, column = 2, sticky = W+E+N+S, padx = 5, pady = 7)
# listData = Listbox(window)
# listData.grid(row = 4, column = 3, padx = 5, pady = 7)

#회원가입 버튼
btnNew = Button(window, text='회원가입', command=clickBtnNew)
btnNew.grid(row = 6, column = 2, sticky = W+E+N+S, padx = 5, pady = 5)

window.mainloop()