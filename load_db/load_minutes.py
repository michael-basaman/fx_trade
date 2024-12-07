import datetime
import psycopg2

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()
cursor2 = conn.cursor()
cursor3 = conn.cursor()

cursor.execute("select start_time, end_time from weeks order by start_time");

records = cursor.fetchall()

count = 0

for record in records:
    start_time = record[0]
    end_time = record[1]

    cursor2.execute("""
      SELECT fx_datetime, bid, ask
        FROM ticks
       WHERE fx_datetime >= %s
         AND fx_datetime < %s
    ORDER BY fx_datetime
    """, (start_time, end_time,))

    records2 = cursor2.fetchall()

    if len(records2) == 0:
        continue

    minutes_array = []
    minutes_dict = {}

    for record2 in records2:
        fx_datetime = record2[0]
        bid = record2[1]
        ask = record2[2]

        minute = fx_datetime.replace(second=0, microsecond=0)

        if minute not in minutes_dict:
            minutes_array.append(minute)
            minutes_dict[minute] = []

        minutes_dict[minute].append(int((float(bid + ask) / 2.0) + 0.01))

    for minute in minutes_array:

        open_price = minutes_dict[minute][0]
        close_price = minutes_dict[minute][-1]
        min_price = open_price
        max_price = close_price

        for price in minutes_dict[minute]:
            if price < min_price:
                min_price = price
            if price > max_price:
                max_price = price

        cursor3.execute("""
        INSERT INTO minutes (fx_datetime, open_price, close_price, min_price, max_price)
        VALUES (%s, %s, %s, %s, %s)
        """, (minute, open_price, close_price, min_price, max_price,))

        count = count + 1

        if count % 1000 == 0:
            print(count)

    conn.commit()





