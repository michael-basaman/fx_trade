import numpy as np
import tensorflow as tf
import time
import psutil
import psycopg2
import math
import os
import shutil
import datetime
# import random

from sklearn.model_selection import train_test_split

EPOCHS = 5000
TEST_SIZE = 0.2


def main():
    model_number = 4
    do_training = True
    manual_save = True
    patience = 50

    start_time = time.time()

    data, labels, class_weight = load_data(model_number)

    print(f"loaded {len(data)} sequences in {format_seconds(time.time() - start_time)}")
    start_time = time.time()

    # seed = random.randint(0, 2 ** 32 - 1)
    seed = 1971504492

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=seed
    )

    print(f"split {len(data)} sequences in {format_seconds(time.time() - start_time)}")
    start_time = time.time()

    checkpoint_path = f"C:/VirtualBox/rsync/fx_trade/checkpoints/{model_number}_{seed}.model.keras"

    if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        print(f"loading model from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        model = get_model(model_number)

    print(f"created model in {format_seconds(time.time() - start_time)}")
    start_time = time.time()

    if do_training:
        # if manual_save:
        #     max_accuracy = 0
        #     no_improvement_count = 0
        #
        #     epoch_number = 0
        #     while True:
        #         epoch_number = epoch_number + 1
        #
        #         model.fit(x_train, y_train,
        #                   epochs=1)
        #
        #         loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        #
        #         if accuracy > max_accuracy:
        #             print(f"Epoch {epoch_number}: accuracy improved from {max_accuracy:.6f} to {accuracy:.6f}, saving model to {checkpoint_path}")
        #             max_accuracy = accuracy
        #             model.save(checkpoint_path)
        #         else:
        #             print(f"Epoch {epoch_number}: accuracy did not improve from {max_accuracy:.6f}")
        #             no_improvement_count + no_improvement_count + 1
        #
        #             if no_improvement_count >= patience:
        #                 print(f"did not improve for {no_improvement_count} epochs, stopping")
        #                 break
        #
        #
        # else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         save_best_only=True,
                                                         save_weights_only=False,
                                                         verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                          mode='max',
                                                          patience=patience,
                                                          verbose=1)

        model.fit(x_train, y_train,
                  epochs=EPOCHS,
                  callbacks=[early_stopping, cp_callback],
                  validation_data=(x_test, y_test))

    print(f"trained model in {format_seconds(time.time() - start_time)}")
    start_time = time.time()

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f"accuracy: {accuracy:.6f}, loss: {loss:.6f}")

    if do_training:
        now = datetime.now()

        model_path = f"C:/VirtualBox/rsync/fx_trade/models/{model_number}_{seed}_{now.strftime("%Y%m%d_%H%M%S")}.model.keras"

        shutil.move(checkpoint_path, model_path)

    print(f"finished in {format_seconds(time.time() - start_time)}")


def load_data(model_number):
    process = psutil.Process()

    conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

    cursor = conn.cursor()
    cursor2 = conn.cursor()

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

    if model_number in {1}:
        timeseries_length = 50
        skip_length = 1
    elif model_number in {2, 3, 4}:
        timeseries_length = 30
        skip_length = 30
    else:
        print(f"load_data() model_number {model_number} not defined")
        exit(1)

    timeseries_length = timeseries_length - 1

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
            minute_index = minute_index + skip_length

        memory_info = process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(f"start_date: {session[0]}, count: {len(labels)}, memory: {array_memory:,}")

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

    return np.array(data_minutes, dtype=np.float32), np.array(labels), class_weight


def get_model(model_number):

    if model_number in {1, 2, 4}:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(3, activation="softmax")
        ])
    elif model_number in {3}:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(512, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(128, dropout=0.2, return_sequences=True),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(3, activation="softmax")
        ])
    else:
        print(f"get_model() model_number {model_number} not defined")
        exit(1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = 'sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    return model


def format_seconds(seconds):
    days = int(math.floor(seconds / 86400))
    seconds -= days * 86400

    hours = int(math.floor(seconds / 3600))
    seconds -= hours * 3600

    minutes = int(math.floor(seconds / 60))
    seconds -= minutes * 60

    seconds = math.floor(seconds)

    if days == 0:
        return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    elif days == 1:
        return '1 day, {:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    else:
        return '{} days, {:02d}:{:02d}:{:02d}'.format(days, hours, minutes, seconds)


# def normalize_array(x_array):
#     input_size = None
#     if len(x_array) > 0:
#         if len(x_array[0]) > 0:
#             input_size = len(x_array[0][0])
#     if input_size is None:
#         return
#
#     for input_i in range(input_size):
#         for window_i in range(len(x_array)):
#             value_sum = 0
#             for minute_i in range(len(x_array[window_i])):
#                 value_sum = value_sum + x_array[window_i][minute_i][input_i]
#             average = value_sum / len(x_array[window_i])
#
#             variance_sum = 0
#             for minute_i in range(len(x_array[window_i])):
#                 variance_sum = variance_sum + ((average - x_array[window_i][minute_i][input_i]) ** 2.0)
#             variance = variance_sum / len(x_array[window_i])
#             stddev = math.sqrt(variance)
#
#             for minute_i in range(len(x_array[window_i])):
#                 x_array[window_i][minute_i][input_i] = (x_array[window_i][minute_i][input_i] - average) / stddev


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
