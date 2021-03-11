## 회원 수정 ##

import pymysql
from tkinter import *
from tkinter import messagebox
import Root

#함수 선언부
def clickBtnEdit2() :
    global window
    global cur, conn
    global gtPw, gtRPw, gtName, getId, user_info

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    getId = Root.edtId1.get()
    gtPw = gtPw.get()
    gtRPw = gtRPw.get()
    gtName = gtName.get()
    sql = "UPDATE user_table SET u_pw = '" + gtPw + "', u_rpw = '" + gtRPw + "', u_name = '" + gtName + "' WHERE u_Id = '"+ getId +"'"
    #sql = "UPDATE user_table SET (u_pw, u_rpw, u_name) =  (str(gtPw), str(gtRPw),str(gtName)) WHERE u_Id = %s"
    cur.execute(sql)

    print(getId)
    print(sql)

    conn.commit()
    cur.close()
    conn.close()
    messagebox.showerror('성공', '정보가 수정되었습니다.')

    if gtPw == NONE or gtPw == "" : # 아무것도 입력하지 않았을때
        messagebox.showerror('오류', '비밀번호를 입력하세요.')
        pass
    elif gtRPw == None or gtRPw == '' :
        messagebox.showerror('오류', '비밀번호 확인을 입력하세요.')
        pass
    elif gtName == None or gtName == '' :
        messagebox.showerror('오류', '닉네임을 입력하세요.')
        pass

    window.destroy()

#전역 변수부
window = None
gtPw, gtRPw, gtName, getId = "","","",""

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
    global window
    global cur, conn
    global gtPw, gtRPw, gtName, user_info,getId

    window = Tk()
    window.geometry("400x400")
    window.title("회원 수정")
    window.configure(background="#8eceff")

    title = Label(window, text="회원 수정", font=("맑은 고딕", 20))
    title.grid(row=0, column=1, sticky=W + E + N + S, pady=10)
    title.configure(background = "#8eceff")

    #아이디
    textId4 = Label(window, text="아이디")
    textId4.grid(row=1, column=1, padx=5, pady=3)
    textId4.configure(background = "#8eceff")

    # 비밀번호
    textPw = Label(window, text="비밀번호")
    textPw.grid(row=2, column=1, padx=5, pady=3)
    textPw.configure(background="#8eceff")

    # 비밀번호 확인
    textRpw = Label(window, text="비밀번호 확인")
    textRpw.grid(row=3, column=1, padx=5, pady=3)
    textRpw.configure(background="#8eceff")

    # 닉네임
    textName = Label(window, text="닉네임")
    textName.grid(row=4, column=1, padx=5, pady=3)
    textName.configure(background="#8eceff")

    # 나이
    textAge = Label(window, text="나이")
    textAge.grid(row=5, column=1, padx=5, pady=3)
    textAge.configure(background="#8eceff")

    # 성별
    textSex = Label(window, text="성별")
    textSex.grid(row=6, column=1, padx=5, pady=3)
    textSex.configure(background="#8eceff")

    # ID 조회
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()

    getId = Root.edtId1.get()
    sql = "SELECT * FROM user_table WHERE user_table.u_Id = %s"
    cur.execute(sql, getId)
    user_info = cur.fetchone()
    print(getId)
    print(sql)
    print(user_info)

    gtId = Label(window, text=user_info[0])
    gtId.grid(row=1, column=3, padx=5, pady=3)
    gtId.configure(background = "#8eceff")

    # 나머지 정보 넣기
    gtPw = Label(window, text=user_info[1])
    gtPw.grid(row=2, column=3, padx=5, pady=3)
    gtPw.configure(background="#8eceff")

    gtRPw = Label(window, text=user_info[2])
    gtRPw.grid(row=3, column=3, padx=5, pady=3)
    gtRPw.configure(background="#8eceff")

    gtName = Label(window, text=user_info[3])
    gtName.grid(row=4, column=3, padx=5, pady=3)
    gtName.configure(background="#8eceff")

    gtAge = Label(window, text=user_info[4])
    gtAge.grid(row=5, column=3, padx=5, pady=3)
    gtAge.configure(background="#8eceff")

    gtSex = Label(window, text=user_info[5])
    gtSex.grid(row=6, column=3, padx=5, pady=3)
    gtSex.configure(background="#8eceff")

    conn.commit()
    cur.close()
    conn.close()

    # 수정할 입력 칸 만들기
    gtPw = Entry(window, width = 10)
    gtPw.grid(row=2, column=4, padx=5, pady=3)

    gtRPw = Entry(window, width = 10)
    gtRPw.grid(row=3, column=4, padx=5, pady=3)

    gtName = Entry(window, width = 10)
    gtName.grid(row=4, column=4, padx=5, pady=3)

    #정보 수정 버튼
    btnEdit = Button(window, text='정보 수정', command=clickBtnEdit2)
    btnEdit.grid(row=7, column=4, sticky=W + E + N + S, padx=5, pady=7)

    window.mainloop()