## 관리자 페이지 ##

from tkinter import *
import pymysql
import User_Search
import User_edit
import Signin
import User_Del

## 함수 선언부
def clickBtnSearch() :
    global edtId1, edtId2, window
    User_Search.main()

def clickBtnNew() :
    global edtId1, edtId2, window
    Signin.main()

def clickBtnEdit() :
    global edtId1, edtId2, window
    User_edit.main()

def clickBtnDel() :
    global edtId1, edtId2, window
    # global cur, conn
    #
    # while True :
    #     # ID와 PW 일치하는지 비교
    #     conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    #     cur = conn.cursor()
    #
    #     tmpId = edtId2.get()
    #     print(tmpId)
    #
    #     sql = "SELECT u_Id FROM user_table"
    #     cur.execute(sql)
    #
    #     u_Id = cur.fetchall()
    #     print(u_Id[0:])
    #
    #     print("('" + tmpId + "',)")
    #     # for data in u_Id, u_Pw :
    #     #     print(data)
    #
    #     #loginInfo = u_Id
    #
    #     if "('" + tmpId + "',)" in u_Id[0:] :
    #         print("로그인 성공")
    #         window.destroy()
    #         User_Del.main()
    #     else:
    #         break
    #
    #     conn.commit()
    #     cur.close()
    #     conn.close()
    if edtId2 != None :
        User_Del.main()
    else :
        pass

# #def clickBtnOut() :
#     global edtId1, edtId2, window
#
#     window.destroy()
#     Login

## 전역 변수부
edtId1, edtId2 = None, None
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
    global edtId1, edtId2, window

    window = Tk()
    window.geometry('400x400')
    window.title("회원 관리 시스템")

    wall_root = PhotoImage(file="C:/images/wall/wall_root.gif")
    wall_label = Label(image=wall_root)
    wall_label.place(x=-2, y=-2)

    text1 = Label(window, text="관리자페이지")
    text1.grid(row = 0, column = 2, sticky = W+E+N+S, pady = 10)
    text1.configure(background = "#8eceff", font=("맑은 고딕", 20))

    # 1. 회원 조회 (조회 후 수정 / 삭제 가능)
    btnSearch = Button(window, text='회원 조회', command=clickBtnSearch)
    btnSearch.grid(row=1, column=3, sticky=W + E + N + S, padx=5, pady=7)

    # 2. 신규 회원 등록
    btnNew = Button(window, text='회원 등록', command=clickBtnNew)
    btnNew.grid(row=2, column=3, sticky=W + E + N + S, padx=5, pady=7)

    # 3. 회원 수정 - 회원 아이디
    textId1 = Label(window, text="회원 수정 - 회원 아이디")
    textId1.grid(row=3, column=2, padx=5, pady=3)
    textId1.configure(background = "#8eceff")

    edtId1 = Entry(window, width=10)
    edtId1.grid(row=3, column=3, padx=5, pady=3)

    btnEdit = Button(window, text='수정', command=clickBtnEdit)
    btnEdit.grid(row=3, column=4, sticky=W + E + N + S, padx=5, pady=7)

    # 4. 회원 삭제 - 회원 아이디
    textId2 = Label(window, text="회원 삭제 - 회원 아이디")
    textId2.grid(row=4, column=2, padx=5, pady=3)
    textId2.configure(background = "#8eceff")

    edtId2 = Entry(window, width=10)
    edtId2.grid(row=4, column=3, padx=5, pady=3)

    btnDel = Button(window, text='삭제', command=clickBtnDel)
    btnDel.grid(row=4, column=4, sticky=W + E + N + S, padx=5, pady=7)

    # 로그아웃 버튼

    # btnLogout = Button(window, text='로그아웃', command=clickBtnOut)
    # btnLogout.grid(row=5, column=5, sticky=W + E + N + S, padx=5, pady=7)

    window.mainloop()