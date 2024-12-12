import numpy as np
import tensorflow as tf
import time
import psutil
import psycopg2
import math
import datetime
import random

# import random
# import json
# import gc
# import io
# import os
# from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split

EPOCHS = 500
TEST_SIZE = 0.4


def normalize_array(x_array):
    input_size = None
    if len(x_array) > 0:
        if len(x_array[0]) > 0:
            input_size = len(x_array[0][0])
    if input_size is None:
        return

    for input_i in range(input_size):
        for window_i in range(len(x_array)):
            value_sum = 0
            for minute_i in range(len(x_array[window_i])):
                value_sum = value_sum + x_array[window_i][minute_i][input_i]
            average = value_sum / len(x_array[window_i])

            variance_sum = 0
            for minute_i in range(len(x_array[window_i])):
                variance_sum = variance_sum + ((average - x_array[window_i][minute_i][input_i]) ** 2.0)
            variance = variance_sum / len(x_array[window_i])
            stddev = math.sqrt(variance)

            for minute_i in range(len(x_array[window_i])):
                x_array[window_i][minute_i][input_i] = (x_array[window_i][minute_i][input_i] - average) / stddev


def main():
    run_time = datetime.datetime.now()

    start_time = time.time()

    data, labels, class_weight = load_data()

    print(f"loaded {len(data)} sequences in {time.time() - start_time} seconds")

    start_time = time.time()

    seed = random.randint(0, 2 ** 32 - 1)
    seed = 1971504492

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=seed
    )

    print(f"split {len(data)} windows in {time.time() - start_time} seconds")

    # database already normalized
    #
    # normalize_array(x_train)
    # normalize_array(x_test)
    # print(f"normalized {len(data)} hours in {time.time() - start_time} seconds")

    checkpoint_path = f"C:/VirtualBox/sourcetree/fx_trade/predict/checkpoints/1_{seed}.weights.h5"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model = get_model()

    start_time = time.time()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50)

    model.fit(x_train, y_train,
              epochs=EPOCHS,
              # class_weight=class_weight,
              callbacks=[early_stopping, cp_callback])

    print(f"fit {len(x_train)} windows in {time.time() - start_time} seconds")

    # result = model.predict(x_test)
    # print(result)
    #
    # if len(result) != len(y_test):
    #     print(f"invalid results size - result: {len(result)}, y_train: {len(y_test)}")
    #     exit()

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"accuracy: {accuracy:.6f}, loss: {loss:.6f}")

    model.save(f"C:/VirtualBox/sourcetree/fx_trade/predict/models/1_{seed}.keras")

    print("finished")


def load_data():
    process = psutil.Process()

    conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

    cursor = conn.cursor()
    cursor2 = conn.cursor()

    start_time = time.time()

    cursor.execute("""
    SELECT start_time, end_time
    FROM sessions
    WHERE complete is true
    order by start_time
    """)

    sessions = cursor.fetchall()

    memory_info = process.memory_info()
    initial_memory = memory_info.rss

    data_minutes = []
    labels = []

    timeseries_length = 50 - 1

    for session in sessions:
        cursor2.execute("""
        SELECT m.fx_datetime, m.sma_26, m.sma_50, m.sma_200, m.macd, m.rsi, m.williams, l.label
        FROM minutes m, labels l
        WHERE m.fx_datetime >= %s
        AND m.fx_datetime < %s
        AND l.pips = 1000
        AND l.fx_datetime = m.fx_datetime
        ORDER BY m.fx_datetime
        """, (session[0], session[1],))

        minutes = cursor2.fetchall()

        minute_index = timeseries_length

        while minute_index < len(minutes):
            window_minutes = []

            for element_index in range(minute_index - timeseries_length, minute_index + 1):
                window_minutes.append(np.array([minutes[element_index][1],
                                                minutes[element_index][2],
                                                minutes[element_index][3],
                                                minutes[element_index][4],
                                                minutes[element_index][5],
                                                minutes[element_index][6],
                                            ], dtype=np.float32))

            data_minutes.append(np.array(window_minutes, dtype=np.float32))
            labels.append(minutes[minute_index][7] + 1) # TODO: fix label data to start with 0
            minute_index = minute_index + 1

        memory_info = process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(session[0], len(labels), array_memory, array_memory / len(labels))

    elapsed1 = time.time() - start_time

    d_count = {}
    for label in labels:
        if label not in d_count:
            d_count[label] = 1
        else:
            d_count[label] = d_count[label] + 1

    total_weights = 0
    for count_label in d_count.keys():
        total_weights = total_weights + d_count[count_label]

    class_weight = {}
    for count_label in [0, 1, 2]:
        if d_count[count_label] == 0:
            weight = 1
        else:
            weight = (1 / d_count[count_label]) * (total_weights / 3)

        class_weight[count_label] = weight

    print(f"class_weights: {class_weight}")

    memory_info = process.memory_info()
    total_memory = memory_info.rss

    print("data loaded", len(data_minutes), elapsed1, total_memory)

    return np.array(data_minutes, dtype=np.float32), np.array(labels), class_weight


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(512, return_sequences=True),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        #tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        # tf.keras.layers.Dropout(0.2),
        # #tf.keras.layers.BatchNormalization(),
        #
        # tf.keras.layers.LSTM(128, return_sequences=True),
        # tf.keras.layers.Dropout(0.2),
        #
        # tf.keras.layers.LSTM(32),
        # tf.keras.layers.Dropout(0.2),
        # #
        # tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = 'sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

# layer_sizes = [16, 32, 64, 128, 256]
# all_layers = [(0,0,)]
# for num_layers in range(2, 5):
#     for hidden_layer_i in range(len(layer_sizes)):
#         last_all_layers = [[(0,0,)]]
#
#         for layer_i in range(num_layers):
#             this_layers = []
#             for last_layer in last_all_layers:
#                 for layer_type in range(1, 4):
#                     if layer_type == 1:
#                         for param_i in range(len(layer_sizes)):
#                             next_layer = []
#                             for last_layer_e in last_layer:
#                                 next_layer.append(last_layer_e)
#                             next_layer.append((1, layer_sizes[hidden_layer_i], layer_sizes[param_i]))
#                             this_layers.append(next_layer)
#                     elif layer_type == 2:
#                         for param_i in range(len(layer_sizes)):
#                             for kernel_size in [2, 4, 8]:
#                                 next_layer = []
#                                 for last_layer_e in last_layer:
#                                     next_layer.append(last_layer_e)
#                                 next_layer.append((2, layer_sizes[hidden_layer_i], layer_sizes[param_i], kernel_size))
#                                 this_layers.append(next_layer)
#
#                             # kernel_size = random.randint(2, 16)
#                             # layers.append((2, layer_sizes[layers_i], kernel_size))
#                     elif layer_type == 3:
#                         for pool_size in [2, 4, 8]:
#                             next_layer = []
#                             for last_layer_e in last_layer:
#                                 next_layer.append(last_layer_e)
#                             next_layer.append((3, layer_sizes[hidden_layer_i], pool_size))
#                             this_layers.append(next_layer)
#             last_all_layers = []
#
#             for this_layer_e in this_layers:
#                 last_all_layers.append(this_layer_e)
#
#         for this_layer_e in this_layers:
#             all_layers.append(this_layer_e)
#
#         print(f"num_layers: {num_layers}, this_layers: {len(this_layers)}")
#         with open(f"C:/VirtualBox/pyworkspace/trade/test_json_{num_layers}_{layer_sizes[hidden_layer_i]}.txt", "w") as file_output:
#             file_output.truncate()
#             file_output.write(json.dumps(this_layers))
#             file_output.write("\n")
#         print(json.dumps(this_layers))
#
# print(f"all_layers: {len(all_layers)}")
