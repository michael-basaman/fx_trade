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

                filename = f"C:/VirtualBox/tickstory/EURUSD/{year_str}/{month_index_str}/{day_str}/{hour_str}h_ticks.bi5"

                if os.path.isfile(filename):
                    if os.stat(filename).st_size == 0:
                        cursor.execute("SELECT 1 FROM empty_hours WHERE filename = %s", (filename,))
                        exists = cursor.fetchone()

                        if exists is not None:
                            print(f"alreaded loaded {filename}")
                            continue

                        hour_datetime_str = f"{date_str} {hour_str}:00:00 UTC"

                        cursor.execute("INSERT INTO empty_hours (fx_datetime, filename) VALUES (%s, %s)",
                            (hour_datetime_str, filename,))
                        conn.commit()
