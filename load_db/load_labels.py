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

start_time = time.time()

cursor.execute("SELECT start_time, end_time FROM weeks WHERE minutes > 0 order by start_time")

weeks = cursor.fetchall()

# log_minute = datetime.datetime.strptime("2003-05-05 03:11:00 -0400", "%Y-%m-%d %H:%M:%S %z")
# found_minute = False

fee = 40

for pips_i in range(5, 6):
    total_count = 0

    pips = pips_i * 100

    buy_count = 0
    neutral_count = 0
    sell_count = 0
    outcomes = []

    for week in weeks:
        count = 0

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
          SELECT start_time, end_time, complete
            FROM sessions
           WHERE start_time > %s
             AND end_time < %s
        ORDER BY start_time
          """, (week[0], week[1],))

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
                # if log_minute == minute[0]:
                #     print(f"found minute {minute[0].strftime("%Y-%m-%d %H:%M:%S")}")
                #     found_minute = True

                while tick_start < len(ticks):
                    if ticks[tick_start][0] < minute[0]:
                        tick_start = tick_start + 1
                        # if found_minute:
                        #     print(f"not yet {ticks[tick_start][0].strftime("%Y-%m-%d %H:%M:%S")}, tick_start: {tick_start}")

                    else:
                        # if found_minute:
                        #     print(f"ready {ticks[tick_start][0].strftime("%Y-%m-%d %H:%M:%S")}, tick_start: {tick_start}")

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

                # if found_minute:
                #     print(f"initial_bid: {initial_bid}, initial_ask: {initial_ask}, tick_index: {tick_index}, tick: {ticks[tick_index][0].strftime("%Y-%m-%d %H:%M:%S")}")

                tick_index = tick_index + 1

                shouldBuy = True
                shouldSell = True
                buyWins = False
                sellWins = False

                while tick_index < len(ticks):
                    buy_profit = ticks[tick_index][1] - initial_ask
                    sell_profit = initial_bid - ticks[tick_index][2]

                    buy_profit = buy_profit - fee
                    sell_profit = sell_profit - fee

                    # if found_minute:
                    #     print(f"buy_profit: {buy_profit}, sell_profit: {sell_profit}, tick_index: {tick_index}, tick: {ticks[tick_index][0].strftime("%Y-%m-%d %H:%M:%S")}")

                    tick_index = tick_index + 1

                    if buy_profit < (-1 * pips):
                        # if found_minute:
                        #     print("setting shouldBuy to false")

                        shouldBuy = False

                    if sell_profit < (-1 * pips):
                        # if found_minute:
                        #     print("setting shouldSell to false")

                        shouldSell = False

                    if not shouldBuy and not shouldSell:
                        # if found_minute:
                        #     print("breaking with shouldBuy and shouldSell both false")

                        break

                    if shouldBuy and buy_profit > pips:
                        # if found_minute:
                        #     print("breaking with buyWins true")

                        buyWins = True
                        break

                    if shouldSell and sell_profit > pips:
                        # if found_minute:
                        #     print("breaking with shouldSell true")

                        sellWins = True
                        break

                if buyWins:
                    label = 1
                elif sellWins:
                    label = -1
                else:
                    label = 0

                tick_index = tick_index - 1
                if tick_index < len(ticks):
                    outcome_seconds = (ticks[tick_index][0] - minute[0]).total_seconds()
                    outcome_seconds = outcome_seconds - 60

                    # if found_minute:
                    #     print(f"outcome_seconds: {outcome_seconds}, {ticks[tick_index][0].strftime("%Y-%m-%d %H:%M:%S")} - {minute[0].strftime("%Y-%m-%d %H:%M:%S")}")
                    #     exit(1)
                else:
                    outcome_seconds = 0

                cursor5.execute("""
                INSERT INTO labels2 (pips, fx_datetime, label, outcome_seconds)
                VALUES (%s, %s, %s, %s)
                """, (pips, minute[0], label, outcome_seconds,))

                if session[2]:
                    if buyWins:
                        buy_count = buy_count + 1
                    elif sellWins:
                        sell_count = sell_count + 1
                    else:
                        neutral_count = neutral_count + 1

                    if outcome_seconds > 0:
                        outcomes.append(outcome_seconds)

                count = count + 1

            conn.commit()

        total_count = total_count + count
        print(f"pips: {pips}, start_time: {week[0]}, end_time: {week[1]}, count: {count}, total_count: {total_count}")

    outcome_count = buy_count + sell_count + neutral_count

    if outcome_count > 0:
        buy_rate = buy_count / outcome_count
        sell_rate = sell_count / outcome_count
        neutral_rate = neutral_count / outcome_count
    else:
        buy_rate = 0
        sell_rate = 0
        neutral_rate = 0

    if len(outcomes) > 0:
        sum_outcome_seconds = 0
        for outcome_seconds in outcomes:
            sum_outcome_seconds = sum_outcome_seconds + outcome_seconds
        avg_outcome_seconds = sum_outcome_seconds / len(outcomes)

        sum_variance = 0
        for outcome_seconds in outcomes:
            sum_variance = sum_variance + ((avg_outcome_seconds - outcome_seconds) ** 2)
        variance = sum_variance / len(outcomes)
        stddev = math.sqrt(variance)
    else:
        avg_outcome_seconds = 0
        stddev = 0

    cursor5.execute("""
    INSERT INTO outcomes (pips, average_seconds, stddev_seconds, buy_rate, neutral_rate, sell_rate)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (pips, avg_outcome_seconds, stddev, buy_rate, neutral_rate, sell_rate,))
    conn.commit()


