
import pymysql
conn=pymysql.connect(host='127.0.0.1', user='root', password='0000', db='soloDB', charset='utf8')

cur =conn.cursor()
# cur.execute("CREATE TABLE userTable (id char(4), userName char(15), email char(20), birthYear int)")
cur.execute("INSERT INTO userTable VALUES( 'hong', '홍지윤', 'hong@naver.com', 1996)")
cur.execute("INSERT INTO userTable VALUES( 'kim', '김태연', 'kim@daum.net', 2011)")
cur.execute("INSERT INTO userTable VALUES( 'star', '별사랑', 'star@daum.net', 1990)")
cur.execute("INSERT INTO userTable VALUES( 'yang', '양지은', 'yang@gmail.com', 1993)")

conn.commit()
conn.close() #연결 끊기