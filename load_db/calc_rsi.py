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

cursor.execute("""
SELECT start_time, end_time
FROM weeks
WHERE minutes > 0
ORDER BY start_time""")

weeks = cursor.fetchall()

all_minutes = []
all_sessions = []

for week in weeks:
    week_minutes = []

    print(week[0])

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
print(f"elapsed_time: {elapsed_time}")

current_weight_26 = 2.0 / 27.0
current_weight_12 = 2.0 / 13.0
previous_weight_26 = 1 - current_weight_26
previous_weight_12 = 1 - current_weight_12

current_smoothing = 1.0 / 14.0
previous_smoothing = 1.0 - current_smoothing

total_count = 0
for week_minutes in all_minutes:
    week_index = 0

    # rsi 14
    # gain = max(0, close - open)
    # loss = max(0, open - close)
    # avg_gain init = sma(14) gain
    # avg_gain next = prev + (1/14) * (close - prev)
    # rsi = 100 * avg_gain / (avg_gain + avg_loss)

    if len(week_minutes) < 15:
        while week_index < len(week_minutes):
            week_minutes[week_index].append(50.0)
            week_index = week_index + 1
        continue

    while week_index < 14:
        week_minutes[week_index].append(50.0)
        week_index = week_index + 1

    if week_index >= len(week_minutes):
        continue

    sum_gain = 0
    sum_loss = 0
    for i in range(week_index - 13, week_index + 1):
        sum_gain = sum_gain + max(0, week_minutes[i][2] - week_minutes[i - 1][2])
        sum_loss = sum_loss + max(0, week_minutes[i - 1][2] - week_minutes[i][2])
    avg_gain = sum_gain / 14.0
    avg_loss = sum_loss / 14.0



    if (avg_gain + avg_loss) == 0:
        week_minutes[week_index].append(50.0)
    else:
        week_minutes[week_index].append(100.0 * avg_gain / (avg_gain + avg_loss))

    last_up = avg_gain
    last_down = avg_loss
    week_index = week_index + 1

    while week_index < len(week_minutes):
        gain = max(0, week_minutes[week_index][2] - week_minutes[week_index - 1][2])
        loss = max(0, week_minutes[week_index - 1][2] - week_minutes[week_index][2])

        up = gain * current_smoothing + last_up * previous_smoothing
        down = loss * current_smoothing + last_down * previous_smoothing

        if (up + down) == 0:
            week_minutes[week_index].append(50.0)
        else:
            week_minutes[week_index].append(100.0 * up / (up + down))

        last_up = up
        last_down = down

        week_index = week_index + 1

elapsed_time = time.time() - start_time
print(f"elapsed_time: {elapsed_time}")

sum_rsi = 0
count_rsi = 0

for i in range(len(all_minutes)):
    for minute in all_minutes[i]:
        in_session = False

        for session in all_sessions[i]:
            if session[0] <= minute[0] < session[1]:
                in_session = True
                break

        if in_session:
            sum_rsi = sum_rsi + minute[5]
            count_rsi = count_rsi + 1

print(f"count_rsi: {count_rsi}")

if count_rsi == 0:
    exit(1)

average_rsi = sum_rsi / count_rsi

print(f"average_rsi: {average_rsi}")

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
            sum_var = sum_var + ((average_rsi - minute[5]) ** 2)
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
print(f"elapsed_time: {elapsed_time}")

update_count = 0
for week_minutes in all_minutes:
    for minute in week_minutes:
        if minute[0] == 0:
            continue

        cursor5.execute("""
        UPDATE minutes
        SET rsi = %s
        WHERE fx_datetime = %s
        """, ((minute[5] - average_rsi) / stddev, minute[0]))

        update_count = update_count + 1

    conn.commit()
    print(f"update_count: {update_count}")

elapsed_time = time.time() - start_time
print(f"elapsed_time: {elapsed_time}")


