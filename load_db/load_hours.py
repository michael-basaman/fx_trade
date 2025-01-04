import psycopg2
import os
import datetime

conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

cursor = conn.cursor()

count = 0

for year in range(2050):
    if year < 2003 or year > 2024:
        continue

    for month in range(15):
        if month < 1 or month > 12:
            continue

        for day in range(35):
            if day < 1 or day > 31:
                continue

            for hour in range(25):
                if hour > 23:
                    continue

                month_index = month - 1

                year_str = "{:04}".format(year)
                month_str = "{:02}".format(month)
                month_index_str = "{:02}".format(month_index)
                day_str = "{:02}".format(day)
                hour_str = "{:02}".format(hour)

                date_str = f"{year_str}-{month_str}-{day_str}"

                filename = f"C:/VirtualBox/tickstory/EURUSD_csv/{year_str}/{month_index_str}/{day_str}/{hour_str}h_ticks.csv"

                if os.path.isfile(filename):
                    cursor.execute("SELECT 1 FROM hours WHERE filename = %s", (filename,))
                    exists = cursor.fetchone()

                    if exists is not None:
                        print(f"alreaded loaded {filename}")
                        continue

                    with open(filename) as filein:
                        sec = -1
                        write = True
                        for line in filein:
                            count = count + 1

                            stripped = line.rstrip()
                            tokens = stripped.split(",")

                            stripped_tokens = [token.strip() for token in tokens]

                            time_tokens = stripped_tokens[0].split(".")

                            second_tokens = time_tokens[0].split(":")
                            minutes = int(second_tokens[1])
                            seconds = int(second_tokens[2])

                            millis = 0
                            if len(time_tokens) == 2:
                                millis = int((int(time_tokens[1]) / 1000) + 0.000001)

                            bid = int((float(stripped_tokens[1]) * 10000.0) + 0.01)
                            ask = int((float(stripped_tokens[3]) * 10000.0) + 0.01)

                            datetime_str = f"{date_str} {hour_str}:{'{:02}'.format(minutes)}:{'{:02}'.format(seconds)}.{'{:03}'.format(millis)} UTC"

                            sql = f"INSERT INTO ticks(fx_datetime, bid, ask) VALUES (%s, %s, %s)"
                            cursor.execute(sql, (datetime_str, bid, ask,))

                        hour_datetime_str = f"{date_str} {hour_str}:00:00 UTC"

                        cursor.execute("INSERT INTO hours (fx_datetime, filename) VALUES (%s, %s)", (hour_datetime_str, filename,))
                        conn.commit()

                        print(filename, count)
