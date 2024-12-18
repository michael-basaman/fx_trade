import psycopg2
import numpy as np
import time
import psutil
import math
import datetime

process = psutil.Process()

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()
cursor2 = conn.cursor()
cursor3 = conn.cursor()
cursor4 = conn.cursor()
cursor5 = conn.cursor()

cursor.execute("SELECT start_time, end_time FROM weeks WHERE minutes > 0 order by start_time")

weeks = cursor.fetchall()

fee = 40

week_count = 0

for outcome_minutes in [30]:
    start_time = time.time()

    outcome_minutes_plus_one = outcome_minutes + 1

    profits = []

    for week in weeks:
        cursor2.execute("""
              SELECT fx_datetime, bid, ask FROM ticks
               WHERE fx_datetime >= %s
                 AND fx_datetime < %s
            ORDER BY fx_datetime
            """, (week[0], week[1],))

        ticks = cursor2.fetchall()

        if len(ticks) == 0:
            continue

        cursor3.execute("""
          SELECT start_time, end_time
            FROM sessions
           WHERE start_time > %s
             AND end_time < %s
             AND complete is true
             AND holiday is false
        ORDER BY start_time
          """, (week[0], week[1],))

        print(week[0])
        week_count = week_count + 1

        sessions = cursor3.fetchall()

        tick_start = 0

        for session in sessions:
            cursor4.execute("""
              SELECT fx_datetime
                FROM minutes
               WHERE fx_datetime >= %s
                 AND fx_datetime < %s
            ORDER BY fx_datetime
            """, (session[0], session[1],))

            minutes = cursor4.fetchall()

            if len(minutes) == 0:
                continue

            for minute in minutes:
                while tick_start < len(ticks):
                    if ticks[tick_start][0] < (minute[0] + datetime.timedelta(minutes=1)):
                        tick_start = tick_start + 1
                    else:
                        break

                if tick_start >= len(ticks):
                    continue
                elif tick_start <= 1:
                    tick_start = 0
                    continue

                tick_index = tick_start
                tick_start = tick_start - 1

                initial_bid = ticks[tick_index][1]
                initial_ask = ticks[tick_index][2]

                tick_index = tick_index + 1

                max_buy_profit = -1000000
                max_sell_profit = -1000000

                while tick_index < len(ticks):
                    if ticks[tick_index][0] > (minute[0] + datetime.timedelta(minutes=outcome_minutes_plus_one)):
                        break

                    buy_profit = ticks[tick_index][1] - initial_ask
                    sell_profit = initial_bid - ticks[tick_index][2]

                    tick_index = tick_index + 1

                    buy_profit = buy_profit - fee
                    sell_profit = sell_profit - fee

                    if buy_profit > max_buy_profit:
                        max_buy_profit = buy_profit

                    if sell_profit > max_sell_profit:
                        max_sell_profit = sell_profit

                max_profit = max(max_buy_profit, max_sell_profit)

                if max_profit > -1000000:
                    profits.append(max_profit)

        if week_count % 52 == 0:
            sum_profit = 0
            for profit in profits:
                sum_profit = sum_profit + profit

            average_profit = sum_profit / len(profits)

            sum_variance = 0
            for profit in profits:
                sum_variance = sum_variance + ((average_profit - profit) ** 2)

            variance = sum_variance / len(profits)
            stddev = math.sqrt(variance)

            print(f"week_count: {week_count}, outcome_minutes: {outcome_minutes}, profits: {len(profits)}, average_profit: {average_profit}, stddev: {stddev}")

    sum_profit = 0
    for profit in profits:
        sum_profit = sum_profit + profit

    average_profit = sum_profit / len(profits)

    sum_variance = 0
    for profit in profits:
        sum_variance = sum_variance + ((average_profit - profit) ** 2)

    variance = sum_variance / len(profits)
    stddev = math.sqrt(variance)

    print(f"outcome_minutes: {outcome_minutes}, profits: {len(profits)}, average_profit: {average_profit}, stddev: {stddev}")

    cursor5.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", (f"profit_{outcome_minutes}_average", average_profit))
    cursor5.execute("INSERT INTO normals (name, value) VALUES (%s, %s)", (f"profit_{outcome_minutes}_stddev", stddev))
    conn.commit()

    elapsed_time = time.time() - start_time
    print(f"elapsed_time: {elapsed_time}")


