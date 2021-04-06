from tkinter import *
from tkinter import messagebox
import pymysql
import tkinter.font

## 함수 선언부
def clickBtnSummit() : # 데이터 입력
    global edtName, edtid, edtpw, edtRpw, edtAge, edtSex
    global window
    global conn, cur
    uID, uPW, uRPW, uName, uSex, uAge = "", "", "", "", "", ""

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    uID = edtid.get()
    uPW = edtpw.get()
    uRPW = edtRpw.get()
    uName = edtName.get()
    uAge = edtAge.get()
    uSex = edtSex.get()

    try :
        sql = "INSERT INTO user_table(u_Id, u_pw, u_rpw, u_name, u_age, u_sex) VALUES('"
        sql += uID + "', '" + uPW + "', '" + uRPW + "', '" + uName + "', " + uAge + ", '" + uSex + "')"
        print(sql)
        cur.execute(sql)

    except :
        messagebox.showerror('오류', '데이터 입력 오류가 발생함')
    else :
        messagebox.showerror('성공', '데이터 입력 성공')

    conn.commit()
    conn.close()

    window.destroy()



def clickBtnCancle() :
    global edtName, edtid, edtpw, edtRpw, edtAge, edtSex
    global window
    global conn, cur
    window.destroy()

## 전역 변수부
edtName, edtid, edtpw, edtRpw, edtAge, edtSex = None, None, None, None, None, None
window = None

## MySQL DB 관련
cur, conn = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None

## 메인코드부
def main() :
    global edtName, edtid, edtpw, edtRpw, edtAge, edtSex
    global window
    window = Tk()
    window.geometry('400x500')
    window.title("영상처리프로그램 회원가입")
    window.configure(background="#8eceff")

    # bg_sign = PhotoImage(file="C:/images/wall_sign.gif")
    # bg_label = Label(image=bg_sign)
    # bg_label.place(x=-2, y=-2)

    # 폰트 설정
    #font = tkinter.font.Font(family="맑은 고딕", size=25)

    title = Label(window, text="[회원가입]")
    title.grid(row = 0, column = 1, sticky = W+E+N+S, pady = 10)
    title.configure(background = "#8eceff", font=("맑은 고딕", 20))

    # 입력할 칸
    textId = Label(window, text="*아이디")
    textId.grid(row=1, column=0, padx=5, pady=3)
    textId.configure(background = "#8eceff")

    edtid = Entry(window, width=10)
    edtid.grid(row = 1, column = 1, padx = 20, pady = 20)

    textpw = Label(window, text="*비밀번호")
    textpw.grid(row=2, column=0, padx=5, pady=3)
    textpw.configure(background = "#8eceff")

    edtpw = Entry(window, width=10)
    edtpw.grid(row = 2, column = 1, padx = 20, pady = 20)

    textRpw = Label(window, text="*비밀번호 확인")
    textRpw.grid(row=3, column=0, padx=5, pady=3)
    textRpw.configure(background = "#8eceff")

    edtRpw = Entry(window, width=10)
    edtRpw.grid(row = 3, column = 1, padx = 20, pady = 20)

    textName = Label(window, text="*닉네임")
    textName.grid(row=4, column=0, padx=5, pady=3)
    textName.configure(background = "#8eceff")

    edtName = Entry(window, width=10)
    edtName.grid(row = 4, column = 1, padx = 20, pady = 20)

    textAge = Label(window, text="나이")
    textAge.grid(row=5, column=0, padx=5, pady=3)
    textAge.configure(background = "#8eceff")

    edtAge = Entry(window, width=10)
    edtAge.grid(row = 5, column = 1, padx = 20, pady = 20)

    textSex = Label(window, text="성별 (남/여)")
    textSex.grid(row=6, column=0, padx=5, pady=3)
    textSex.configure(background = "#8eceff")

    edtSex = Entry(window, width=10)
    edtSex.grid(row = 6, column = 1, padx = 20, pady = 20)

    #버튼
    btnSummit = Button(window, text='회원가입', command=clickBtnSummit)
    btnSummit.grid(row=7, column=1, sticky=W + E + N + S, padx=5, pady=7)

    btnCancle = Button(window, text='취소', command=clickBtnCancle)
    btnCancle.grid(row = 7, column = 2, sticky = W+E+N+S, padx = 5, pady = 7)

    window.mainloop()
