# golden cross sma 50 - sma 200
import math

# macd 26 period ema
#      12 period ema
#      9  period ema of (12 period ema - 26 period ema)

# rsi 14
# gain = max(0, close - open)
# loss = max(0, open - close)
# avg_gain init = sma(14) gain
# avg_gain next = prev + (1/14) * (close - prev)
# rsi = 100 * avg_gain / (avg_gain + avg_loss)

# bollinger bands
# sma 20
# std 20
# sma 20 - 2 * std > close > sma 20 + 2 * std

# willians
# R = -100 * (high(n) - close) / (high(n) - low(n)

# sma 50 - norm_all(close - sma 50)
# sma 200 - norm_all(close - sma 200)
# macd - norm_all(macd)
# rsi - norm_all(rsi)
# bollinger - (close - sma20) / std20
# williams - norm_all(williams)

import psycopg2
import datetime
import time

start_time = time.time()

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()
cursor2 = conn.cursor()
cursor3 = conn.cursor()
cursor4 = conn.cursor()
cursor5 = conn.cursor()
cursor6 = conn.cursor()

cursor.execute("""
SELECT start_time, end_time
FROM weeks
WHERE minutes > 0
ORDER BY start_time""")

weeks = cursor.fetchall()

for indicator_length in [26, 50, 200, 15, 30, 60, 120, 240]:
    all_minutes = []
    all_sessions = []

    for week in weeks:
        week_minutes = []

        print(indicator_length, week[0])

        cursor4.execute("""
        SELECT start_time, end_time
        FROM sessions
        WHERE start_time >= %s
        AND end_time < %s
        AND complete is true
        """, (week[0], week[1],))

        sessions = []

        this_sessions = cursor4.fetchall()

        for session in this_sessions:
            sessions.append([session[0], session[1]])

        cursor2.execute("""
        SELECT fx_datetime, open_price, close_price, min_price, max_price
        FROM minutes
        WHERE fx_datetime >= %s
        and fx_datetime < %s
        order by fx_datetime
        """, (week[0], week[1]))

        minutes = cursor2.fetchall()

        last_minute = None
        for minute in minutes:
            current_datetime = minute[0]

            if last_minute is None:
                first_minute = [week[0], minute[1], minute[2], minute[3], minute[4]]
                last_minute = None

                while first_minute[0] < minute[0]:
                    week_minutes.append(first_minute)
                    last_minute = first_minute
                    first_minute = [first_minute[0] + datetime.timedelta(minutes=1), minute[1], minute[2], minute[3], minute[4]]

            while minute[0] - last_minute[0] > datetime.timedelta(minutes=1):
                last_minute = [last_minute[0] + datetime.timedelta(minutes=1), last_minute[1], last_minute[2], last_minute[3], last_minute[4]]
                week_minutes.append(last_minute)

            last_minute = [minute[0], minute[1], minute[2], minute[3], minute[4]]
            week_minutes.append(last_minute)

        all_minutes.append(week_minutes)
        all_sessions.append(sessions)

    elapsed_time = time.time() - start_time
    print(f"load minutes - elapsed_time: {elapsed_time}")

    indicator_length_minus_one = indicator_length - 1

    total_count = 0
    for week_minutes in all_minutes:
        week_index = 0

        if len(week_minutes) < indicator_length:
            while week_index < len(week_minutes):
                week_minutes[week_index].append(0)
                week_minutes[week_index].append(0)
                week_index = week_index + 1
            continue

        while week_index < indicator_length_minus_one:
            week_minutes[week_index].append(0)
            week_minutes[week_index].append(0)
            week_index = week_index + 1

        while week_index < len(week_minutes):
            sum = 0
            for i in range(week_index - indicator_length_minus_one, week_index + 1):
                sum = sum + week_minutes[i][2]
            avg = sum / float(indicator_length)

            week_minutes[week_index].append(avg)
            week_minutes[week_index].append(week_minutes[week_index][2] - avg)
            week_index = week_index + 1

    elapsed_time = time.time() - start_time
    print(f"compute sma - elapsed_time: {elapsed_time}")

    sum_sma = 0
    count_sma = 0

    for i in range(len(all_minutes)):
        for minute in all_minutes[i]:
            in_session = False

            for session in all_sessions[i]:
                if session[0] <= minute[0] < session[1]:
                    in_session = True
                    break

            if in_session:
                sum_sma = sum_sma + minute[6]
                count_sma = count_sma + 1

    print(f"count_sma: {count_sma}")

    if count_sma == 0:
        exit(1)

    average_sma = sum_sma / count_sma

    print(f"average_sma: {average_sma}")

    sum_var = 0
    count_var = 0

    for i in range(len(all_minutes)):
        for minute in all_minutes[i]:
            in_session = False

            for session in all_sessions[i]:
                if session[0] <= minute[0] < session[1]:
                    in_session = True
                    break

            if in_session:
                sum_var = sum_var + ((average_sma - minute[6]) ** 2)
                count_var = count_var + 1

    print(f"count_var: {count_var}")

    if count_var == 0:
        exit(1)

    var = sum_var / count_var

    print(f"count_var: {count_var}")

    stddev = math.sqrt(var)

    print(f"stddev: {stddev}")

    if stddev == 0:
        exit(1)

    elapsed_time = time.time() - start_time
    print(f"compute average, stddev - elapsed_time: {elapsed_time}")

    update_count = 0
    for week_minutes in all_minutes:
        for minute in week_minutes:
            cursor5.execute(f"""
            UPDATE minutes
            SET sma_{indicator_length} = %s
            WHERE fx_datetime = %s
            """, ((minute[6] - average_sma) / stddev, minute[0]))

            update_count = update_count + 1

        conn.commit()
        print(f"update_count: {update_count}")

    cursor6.execute("INSERT INTO indicators (name, average, stddev) VALUES (%s, %s, %s)",
                    (f"sma_{indicator_length}", average_sma, stddev,))
    conn.commit()

    elapsed_time = time.time() - start_time
    print(f"elapsed_time: {elapsed_time}")


