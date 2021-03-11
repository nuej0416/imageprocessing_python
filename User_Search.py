## 회원 조회 ##

import pymysql
from tkinter import *
from tkinter import messagebox
from tkinter import Entry

#함수 선언부
def idSelectData() : # ID 입력해서 조회
    global edt1, btnSelect, btnAllSelect, listData1, listData2, listData3, listData4, listData5, listData6, window
    global conn, cur
    strData1, strData2, strData3, strData4,strData5, strData6 = [], [], [], [], [], []
    stdID1 = ""

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    stdID1 = edt1.get()
    print(stdID1)

    sql = "SELECT * FROM user_table WHERE u_Id = %s"
    cur.execute(sql, stdID1)
    print(sql)

    if stdID1 == NONE or stdID1 == "" : # 아무것도 입력하지 않았을때
        messagebox.showerror('오류', '데이터 조회 오류가 발생함')
        pass

    strData1.append("아이디")
    strData2.append("비밀번호")
    strData3.append("비밀번호 확인")
    strData4.append("닉네임")
    strData5.append("나이")
    strData6.append("성별")

    strData1.append("-------------")
    strData2.append("-------------")
    strData3.append("-------------")
    strData4.append("-------------")
    strData5.append("-------------")
    strData6.append("-------------")

    while (True):
        row = cur.fetchone()
        if row == None:
            break
        strData1.append(row[0])
        strData2.append(row[1])
        strData3.append(row[2])
        strData4.append(row[3])
        strData5.append(row[4])
        strData6.append(row[5])

    listData1.delete(0, listData1.size() - 1)
    listData2.delete(0, listData2.size() - 1)
    listData3.delete(0, listData3.size() - 1)
    listData4.delete(0, listData4.size() - 1)
    listData5.delete(0, listData5.size() - 1)
    listData6.delete(0, listData6.size() - 1)

    for item1, item2, item3, item4, item5, item6 in zip(strData1, strData2, strData3, strData4, strData5,
                                                               strData6):
        listData1.insert(END, item1)
        listData2.insert(END, item2)
        listData3.insert(END, item3)
        listData4.insert(END, item4)
        listData5.insert(END, item5)
        listData6.insert(END, item6)
    #except :
        #messagebox.showerror('오류', '데이터 조회 오류가 발생함')
    # else :  # 활성화 시 데이터를 입력하지 않아도 성공했다는 메세지가 뜸
    #     messagebox.showerror('성공', '데이터 조회 성공')

    conn.close()

def allSelectData() : #전체 학생 테이블 조회
    global edt1, btnSelect, btnAllSelect, listData1, listData2, listData3, listData4, listData5, listData6, window
    global conn, cur
    strData1, strData2, strData3, strData4, strData5, strData6 = [], [], [], [], [], []

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    cur.execute("SELECT * FROM user_table")

    strData1.append("아이디")
    strData2.append("비밀번호")
    strData3.append("비밀번호 확인")
    strData4.append("닉네임")
    strData5.append("나이")
    strData6.append("성별")

    strData1.append("-------------")
    strData2.append("-------------")
    strData3.append("-------------")
    strData4.append("-------------")
    strData5.append("-------------")
    strData6.append("-------------")

    while (True) :
        row = cur.fetchone()
        if row == None :
            break
        strData1.append(row[0])
        strData2.append(row[1])
        strData3.append(row[2])
        strData4.append(row[3])
        strData5.append(row[4])
        strData6.append(row[5])

    listData1.delete(0, listData1.size() - 1)
    listData2.delete(0, listData2.size() - 1)
    listData3.delete(0, listData3.size() - 1)
    listData4.delete(0, listData4.size() - 1)
    listData5.delete(0, listData5.size() - 1)
    listData6.delete(0, listData6.size() - 1)

    for item1, item2, item3, item4, item5, item6 in zip(strData1, strData2, strData3, strData4, strData5, strData6) :
        listData1.insert(END, item1)
        listData2.insert(END, item2)
        listData3.insert(END, item3)
        listData4.insert(END, item4)
        listData5.insert(END, item5)
        listData6.insert(END, item6)

    conn.close()

#전역 변수부
btnSelect, btnAllSelect, listData1, listData2, listData3, listData4, listData5, listData6 = None, None, None, None, None, None,None, None
window = None
stdID1, edt1 = "", ''
## MySQL DB 관련
cur, conn = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None

#메인 코드부

#화면 구성
def main() :
    global edt1, btnSelect, btnAllSelect, listData1, listData2, listData3, listData4, listData5, listData6, window
    window = Tk()
    window.geometry("1000x300")
    window.title("회원 조회")
    window.configure(background="#8eceff")

    edtFrame = Frame(window)
    edtFrame.pack()
    edtFrame.configure(background = "#8eceff")
    listFrame = Frame(window)
    listFrame.pack(side = BOTTOM, fill = BOTH, expand =1)

    #입력할 칸
    edt1 = Entry(edtFrame, width = 10)
    edt1.pack(side = LEFT, padx = 10, pady = 10)

    #버튼
    btnSelect = Button(edtFrame, text = "ID 조회", command = idSelectData)
    btnSelect.pack(side = LEFT, padx = 10, pady = 10)
    btnAllSelect = Button(edtFrame, text = "전체 조회", command = allSelectData)
    btnAllSelect.pack(side = LEFT, padx = 10, pady = 10)

    #보여줄 목록
    listData1 = Listbox(listFrame,bg = "#c6e6ff")
    listData1.pack(side = LEFT, fill = BOTH, expand = 1)
    listData2 = Listbox(listFrame,bg = "#a5d8ff")
    listData2.pack(side = LEFT, fill = BOTH, expand = 1)
    listData3 = Listbox(listFrame,bg = "#c6e6ff")
    listData3.pack(side = LEFT, fill = BOTH, expand = 1)
    listData4 = Listbox(listFrame,bg = "#a5d8ff")
    listData4.pack(side = LEFT, fill = BOTH, expand = 1)
    listData5 = Listbox(listFrame,bg = "#c6e6ff")
    listData5.pack(side = LEFT, fill = BOTH, expand = 1)
    listData6 = Listbox(listFrame,bg = "#a5d8ff")
    listData6.pack(side = LEFT, fill = BOTH, expand = 1)

    window.mainloop()