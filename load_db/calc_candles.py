# golden cross sma 50 - sma 200
import math
import psycopg2
import time

start_time = time.time()

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()
cursor2 = conn.cursor()
cursor3 = conn.cursor()
cursor4 = conn.cursor()
cursor5 = conn.cursor()

cursor.execute("""
    SELECT start_time, end_time
    FROM sessions
    WHERE complete is true
    order by start_time
    """)

sessions = cursor.fetchall()

sum_candle_length = 0
sum_max_length = 0
sum_min_length = 0
minute_count = 0

for session in sessions:
    cursor2.execute("""
    SELECT fx_datetime, open_price, close_price, min_price, max_price
    FROM minutes
    WHERE fx_datetime >= %s
    and fx_datetime < %s
    order by fx_datetime
    """, (session[0], session[1]))

    minutes = cursor2.fetchall()

    for minute in minutes:
        candle_length = minute[2] - minute[1]
        max_length = minute[4] - minute[2]
        min_length = minute[2] - minute[3]

        sum_candle_length = sum_candle_length + candle_length
        sum_max_length = sum_max_length + max_length
        sum_min_length = sum_min_length + min_length

    minute_count = minute_count + len(minutes)

average_candle_length = sum_candle_length / minute_count
average_max_length = sum_max_length / minute_count
average_min_length = sum_min_length / minute_count

sum_candle_variance = 0
sum_max_variance = 0
sum_min_variance = 0

for session in sessions:
    cursor2.execute("""
    SELECT fx_datetime, open_price, close_price, min_price, max_price
    FROM minutes
    WHERE fx_datetime >= %s
    and fx_datetime < %s
    order by fx_datetime
    """, (session[0], session[1]))

    minutes = cursor2.fetchall()

    for minute in minutes:
        candle_length = minute[2] - minute[1]
        max_length = minute[4] - minute[2]
        min_length = minute[2] - minute[3]

        sum_candle_variance = sum_candle_variance + ((average_candle_length - candle_length) ** 2)
        sum_max_variance = sum_max_variance + ((average_max_length - max_length) ** 2)
        sum_min_variance = sum_min_variance + ((average_min_length - min_length) ** 2)

candle_variance = sum_candle_variance / minute_count
max_variance = sum_max_variance / minute_count
min_variance = sum_min_variance / minute_count

candle_stddev = math.sqrt(candle_variance)
max_stddev = math.sqrt(max_variance)
min_stddev = math.sqrt(min_variance)

print(f"average_candle_length: {average_candle_length}, candle_stddev: {candle_stddev}")
print(f"average_max_length: {average_max_length}, max_stddev: {max_stddev}")
print(f"average_min_length: {average_min_length}, min_stddev: {min_stddev}")

elapsed_time = time.time() - start_time
print(f"elapsed_time: {elapsed_time}")

cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", ('candle_average', average_candle_length,))
cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", ('max_average', average_max_length,))
cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", ('min_average', average_min_length,))
cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", ('candle_stddev', candle_stddev,))
cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", ('max_stddev', max_stddev,))
cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", ('min_stddev', min_stddev,))

conn.commit()


elapsed_time = time.time() - start_time
print(f"elapsed_time: {elapsed_time}")
