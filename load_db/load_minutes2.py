import datetime
import psycopg2

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()
cursor2 = conn.cursor()
cursor3 = conn.cursor()

cursor.execute("select start_time, end_time from weeks order by start_time");

records = cursor.fetchall()

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

        minutes_dict[minute].append((bid, ask))

    for minute in minutes_array:
        open_bid = minutes_dict[minute][0][0]
        open_ask = minutes_dict[minute][0][1]
        close_bid = minutes_dict[minute][-1][0]
        close_ask = minutes_dict[minute][-1][1]
        min_bid = open_bid
        min_ask = open_ask
        max_bid = open_bid
        max_ask = open_ask

        for bid, ask in minutes_dict[minute]:
            if bid < min_bid:
                min_bid = bid
            if ask < min_ask:
                min_ask = ask
            if bid > max_bid:
                max_bid = bid
            if ask > max_ask:
                max_ask = ask

        cursor3.execute("""
        INSERT INTO minutes2 (fx_datetime, open_bid, open_ask, close_bid, close_ask, min_bid, min_ask, max_bid, max_ask, ticks)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (minute, open_bid, open_ask, close_bid, close_ask, min_bid, min_ask, max_bid, max_ask, len(minutes_dict[minute])))

    conn.commit()

    print(start_time)





