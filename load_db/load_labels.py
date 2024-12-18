import psycopg2
import time
import psutil
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

fee = 40

for pips in [850]:
    total_count = 0

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
          SELECT start_time, end_time
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

                shouldBuy = True
                shouldSell = True
                buyWins = False
                sellWins = False

                while tick_index < len(ticks):
                    buy_profit = ticks[tick_index][1] - initial_ask
                    sell_profit = initial_bid - ticks[tick_index][2]

                    buy_profit = buy_profit - fee
                    sell_profit = sell_profit - fee

                    tick_index = tick_index + 1

                    if buy_profit < (-1 * pips):
                        shouldBuy = False

                    if sell_profit < (-1 * pips):
                        shouldSell = False

                    if not shouldBuy and not shouldSell:
                        break

                    if shouldBuy and buy_profit > pips:
                        buyWins = True
                        break

                    if shouldSell and sell_profit > pips:
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
                else:
                    outcome_seconds = 0

                cursor5.execute("""
                INSERT INTO labels (pips, fx_datetime, label, outcome_seconds)
                VALUES (%s, %s, %s, %s)
                """, (pips, minute[0], label, outcome_seconds,))

                count = count + 1

            conn.commit()

        total_count = total_count + count
        print(f"pips: {pips}, start_time: {week[0]}, end_time: {week[1]}, count: {count}, total_count: {total_count}")
