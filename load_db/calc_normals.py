# golden cross sma 50 - sma 200
import math
import psycopg2
import time

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

for sma in [15, 30, 60]:
    start_time = time.time()

    sum_variance = 0
    variance_count = 0

    for session in sessions:
        cursor2.execute("""
        SELECT fx_datetime, close_price
        FROM minutes
        WHERE fx_datetime >= %s
        and fx_datetime < %s
        order by fx_datetime
        """, (session[0], session[1]))

        minutes = cursor2.fetchall()

        sma_minus_one = sma - 1
        minute_index = sma_minus_one

        while minute_index < len(minutes):
            sum_close_price = 0

            for element_index in range(minute_index - sma_minus_one, minute_index + 1):
                sum_close_price = sum_close_price + minutes[element_index][1]

            average_close_price = sum_close_price / sma

            for element_index in range(minute_index - sma_minus_one, minute_index + 1):
                sum_variance = sum_variance + ((average_close_price - minutes[element_index][1]) ** 2)

            variance_count = variance_count + sma
            minute_index = minute_index + 1

    variance = sum_variance / variance_count
    stddev = math.sqrt(variance)

    print(f"sma_{sma}_stddev: {stddev}")

    cursor3.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", (f"sma_{sma}_stddev", stddev,))
    conn.commit()

    elapsed_time = time.time() - start_time
    print(f"elapsed_time: {elapsed_time}")
