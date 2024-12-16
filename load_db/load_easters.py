import datetime
import psycopg2

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()

with open("C:/Users/state/Downloads/easter500.txt") as file:
    for line in file:
        tokens = line.strip().split()

        if len(tokens) != 3:
            continue

        if 2000 <= int(tokens[2]) < 2100:
            date = f"{tokens[2]}-{int(tokens[0]):02d}-{int(tokens[1]):02d}"

            easter = datetime.datetime.strptime(date, "%Y-%m-%d")
            friday = easter - datetime.timedelta(days=2)
            monday = easter + datetime.timedelta(days=1)

            cursor.execute("INSERT INTO easters (sunday_date, friday_date, monday_date) VALUES (%s, %s, %s)", (easter.date(),friday.date(), monday.date()))

conn.commit()
