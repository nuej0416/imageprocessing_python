## 회원 삭제 ##

import pymysql
import Root
from tkinter import *
from tkinter import messagebox

#함수 선언부
def clickBtnDelete() :
    global window
    global cur, conn

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    getId = Root.edtId2.get()
    sql = "DELETE FROM user_table WHERE user_table.u_Id = '" + getId + "'"
    cur.execute(sql)
    print(sql)

    conn.commit()
    cur.close()
    conn.close()

    messagebox.showerror('성공', '삭제되었습니다.')
#전역 변수부

#메인 코드부
window = None

## MySQL DB 관련
cur, conn = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None

#화면 구성
def main() :
    global window
    global cur, conn

    window = Tk()
    window.geometry("400x200")
    window.title("회원 삭제")
    window.configure(background="#8eceff")

    title = Label(window, text = "회원 삭제")
    title.grid(row = 0, column = 2, sticky = W+E+N+S, pady = 10)
    title.configure(background = "#8eceff", font=("맑은 고딕", 20))

    # 삭제 아이디
    textId3 = Label(window, text="아이디")
    textId3.grid(row=1, column=2, padx=5, pady=3)
    textId3.configure(background = "#8eceff")

    # ID 조회
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    getId = Root.edtId2.get()
    sql = "SELECT u_Id FROM user_table WHERE user_table.u_Id = '" + getId + "'"
    cur.execute(sql)

    gtId = Label(window, text = getId)
    gtId.grid(row=1, column=3, padx=5, pady=3)
    gtId.configure(background = "#8eceff")

    conn.commit()
    cur.close()
    conn.close()

    # 삭제 닉네임
    textId4 = Label(window, text="닉네임")
    textId4.grid(row=2, column=2, padx=5, pady=3)
    textId4.configure(background = "#8eceff")

    # 닉네임 조회
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    getName = Root.edtId2.get()
    sql = "SELECT u_name FROM user_table WHERE user_table.u_Id = '" + getName + "'"
    cur.execute(sql)

    for getName in cur :
        print(getName)

    gtName = Label(window, text=getName)
    gtName.grid(row=2, column=3, padx=5, pady=3)
    gtName.configure(background = "#8eceff")

    conn.commit()
    cur.close()
    conn.close()

    # 위 회원을 삭제하시겠습니까?
    delLabel = Label(window, text = "위 회원을 삭제하시겠습니까?")
    delLabel.grid(row=3, column=2, padx=5, pady=3)
    delLabel.configure(background = "#8eceff")
    btnDelete = Button(window, text='삭제', command=clickBtnDelete)
    btnDelete.grid(row=3, column=4, sticky=W + E + N + S, padx=5, pady=7)

    window.mainloop()
