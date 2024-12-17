import numpy as np
import tensorflow as tf
import time
import psutil
import psycopg2
import math
#import os
import shutil
import datetime
#import random

#from sklearn.model_selection import train_test_split

process = psutil.Process()


def main():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S")

    timeseries_length = 30
    skip_length = 1
    outcome_minutes = 15
    pips = 500
    patience = 100

    start_time = time.time()

    # data, labels, class_weight = load_data(timeseries_length, skip_length, outcome_minutes, pips)
    #
    # print(f"loaded {len(data)} sequences in {format_seconds(time.time() - start_time)}")
    # start_time = time.time()
    #
    # x_train, x_test, y_train, y_test = train_test_split(
    #     data, labels, test_size=TEST_SIZE
    # )
    #
    # print(f"split {len(data)} sequences in {format_seconds(time.time() - start_time)}")

    memory_info = process.memory_info()
    initial_memory = memory_info.rss

    train_data, train_labels, val_data, val_labels, test_data, test_labels, class_weight = get_data(timeseries_length, skip_length, outcome_minutes, pips)

    memory_info = process.memory_info()
    array_memory = memory_info.rss - initial_memory
    print(
        f"Loaded {len(train_data) + len(val_data) + len(test_data)} minutes in {format_seconds(time.time() - start_time)}, memory: {array_memory:,}")

    if patience > 0:
        exit(1)

    for flatten_units in [128, 64, 0]:
        for layer_count in [3, 2]:
            for base_units in [128, 64, 32, 256]:
                for recurrent_dropout_percent in [10, 0]:
                    for dropout_percent in [20, 0]:
                        for weight_decay in [1, 10, 0]:
                            start_time = time.time()

                            checkpoint_path = f"C:/VirtualBox/rsync/fx_trade/checkpoints/{now_str}_c{layer_count}_u{base_units:03d}_f{flatten_units:03d}_d{dropout_percent:02d}_r{recurrent_dropout_percent:02d}_w{weight_decay:02d}.model.keras"

                            tf.keras.backend.clear_session()
                            model = get_model(layer_count, base_units, flatten_units, dropout_percent, recurrent_dropout_percent, weight_decay)

                            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                             monitor='val_loss',
                                                                             mode='min',
                                                                             save_best_only=True,
                                                                             save_weights_only=False,
                                                                             verbose=1)

                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                              mode='min',
                                                                              patience=patience,
                                                                              restore_best_weights=True,
                                                                              start_from_epoch=patience,
                                                                              verbose=1)

                            model.fit(train_data, train_labels,
                                      epochs=10000,
                                      callbacks=[early_stopping, cp_callback],
                                      class_weight=class_weight,
                                      batch_size=64,
                                      validation_data=(val_data, val_labels))

                            print(f"trained model in {format_seconds(time.time() - start_time)}")
                            start_time = time.time()

                            loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)

                            print(f"accuracy: {accuracy:.6f}, loss: {loss:.6f}")

                            model_path = f"C:/VirtualBox/rsync/fx_trade/models/{now_str}_l{int(10000 * loss):05d}_a{int(10000 * accuracy):05d}_c{layer_count}_u{base_units:03d}_f{flatten_units:03d}_d{dropout_percent:02d}_r{recurrent_dropout_percent:02d}_w{weight_decay:02d}.model.keras"
                            shutil.move(checkpoint_path, model_path)

    print(f"finished in {format_seconds(time.time() - start_time)}")


def get_data(timeseries_length, skip_length, outcome_minutes, pips):
    memory_info = process.memory_info()
    initial_memory = memory_info.rss

    data_840, labels_840, data_600, labels_600 = load_data(timeseries_length, skip_length, outcome_minutes, pips)

    d_count = {}
    for session_labels in labels_840:
        for label in session_labels:
            if label not in d_count:
                d_count[label] = 1
            else:
                d_count[label] = d_count[label] + 1

    for session_labels in labels_600:
        for label in session_labels:
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

    print(f"class_weight: {class_weight}")

    indices = np.random.permutation(len(data_840))

    data_840 = data_840[indices]
    labels_840 = labels_840[indices]

    indices = np.random.permutation(len(data_600))

    data_600 = data_600[indices]
    labels_600 = labels_600[indices]

    split_index = int(0.8 * len(data_840))

    nontest_data_840 = data_840[:split_index]
    nontest_labels_840 = labels_840[:split_index]
    test_data_840 = data_840[split_index:]
    test_labels_840 = labels_840[split_index:]

    split_index = int(0.8 * len(data_600))

    nontest_data_600 = data_600[:split_index]
    nontest_labels_600 = labels_600[:split_index]
    test_data_600 = data_600[split_index:]
    test_labels_600 = labels_600[split_index:]

    split_index = int(0.8 * len(nontest_data_840))

    train_data_840 = nontest_data_840[:split_index]
    train_labels_840 = nontest_labels_840[:split_index]
    val_data_840 = nontest_data_840[split_index:]
    val_labels_840 = nontest_labels_840[split_index:]

    split_index = int(0.8 * len(nontest_data_600))

    train_data_600 = nontest_data_600[:split_index]
    train_labels_600 = nontest_labels_600[:split_index]
    val_data_600 = nontest_data_600[split_index:]
    val_labels_600 = nontest_labels_600[split_index:]

    train_data_840 = np.reshape(train_data_840, (train_data_840.shape[0] * train_data_840.shape[1], train_data_840.shape[2]))
    train_labels_840 = np.reshape(train_labels_840, (train_labels_840.shape[0] * train_labels_840.shape[1], train_labels_840.shape[2]))
    val_data_840 = np.reshape(val_data_840, (val_data_840.shape[0] * val_data_840.shape[1], val_data_840.shape[2]))
    val_labels_840 = np.reshape(val_labels_840, (val_labels_840.shape[0] * val_labels_840.shape[1], val_labels_840.shape[2]))
    test_data_840 = np.reshape(test_data_840, (test_data_840.shape[0] * test_data_840.shape[1], test_data_840.shape[2]))
    test_labels_840 = np.reshape(test_labels_840, (test_labels_840.shape[0] * test_labels_840.shape[1], test_labels_840.shape[2]))

    train_data_600 = np.reshape(train_data_600, (train_data_600.shape[0] * train_data_600.shape[1], train_data_600.shape[2]))
    train_labels_600 = np.reshape(train_labels_600, (train_labels_600.shape[0] * train_labels_600.shape[1], train_labels_600.shape[2]))
    val_data_600 = np.reshape(val_data_600, (val_data_600.shape[0] * val_data_600.shape[1], val_data_600.shape[2]))
    val_labels_600 = np.reshape(val_labels_600, (val_labels_600.shape[0] * val_labels_600.shape[1], val_labels_600.shape[2]))
    test_data_600 = np.reshape(test_data_600, (test_data_600.shape[0] * test_data_600.shape[1], test_data_600.shape[2]))
    test_labels_600 = np.reshape(test_labels_600, (test_labels_600.shape[0] * test_labels_600.shape[1], test_labels_600.shape[2]))

    train_data = np.append(train_data_840, train_data_600)
    train_labels = np.append(train_labels_840, train_labels_600)
    val_data = np.append(val_data_840, val_data_600)
    val_labels = np.append(val_labels_840, val_labels_600)
    test_data = np.append(test_data_840, test_data_600)
    test_labels = np.append(test_labels_840, test_labels_600)

    indices = np.random.permutation(len(train_data))

    train_data = train_data[indices]
    train_labels = train_labels[indices]

    indices = np.random.permutation(len(val_data))

    val_data = val_data[indices]
    val_labels = val_labels[indices]

    indices = np.random.permutation(len(test_data))

    test_data = test_data[indices]
    test_labels = test_labels[indices]

    memory_info = process.memory_info()
    array_memory = memory_info.rss - initial_memory
    print(f"get_data() - train_data: {len(train_data)}, val_data: {len(val_data)}, test_data: {len(test_data)}, memory: {array_memory:,}")

    return (train_data, train_labels, val_data, val_labels, test_data,
            test_labels, class_weight)


def load_data(timeseries_length, skip_length, outcome_minutes, pips):
    conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

    cursor = conn.cursor()
    cursor2 = conn.cursor()

    cursor.execute("""
    SELECT start_time, end_time
    FROM sessions
    WHERE complete is true
    AND holiday is false
    order by start_time
    """)

    sessions = cursor.fetchall()

    memory_info = process.memory_info()
    initial_memory = memory_info.rss

    data = []
    labels = []

    outcome_seconds = outcome_minutes * 60

    for session in sessions:
        cursor2.execute("""
        SELECT m.fx_datetime, m.close_price, m.macd, m.sma_60, m.macd, m.rsi, m.williams,
               case when l.outcome_seconds > %s then 0
               else l.label end outcome_label
        FROM minutes m, labels2 l
        WHERE m.fx_datetime >= %s
        AND m.fx_datetime < %s
        AND l.pips = %s
        AND l.fx_datetime = m.fx_datetime
        ORDER BY m.fx_datetime
        """, (outcome_seconds, session[0], session[1], pips,))

        minutes = cursor2.fetchall()

        if len(minutes) not in {600, 840}:
            continue

        timeseries_length_minus_one = timeseries_length - 1
        minute_index = timeseries_length_minus_one

        session_data = []
        session_labels = []

        while minute_index < len(minutes):
            window_minutes = []

            sum_price = 0
            for element_index in range(minute_index - timeseries_length_minus_one, minute_index + 1):
                sum_price = sum_price + minutes[element_index][1]
            avg_price = sum_price / timeseries_length
            sum_variance = 0
            for element_index in range(minute_index - timeseries_length_minus_one, minute_index + 1):
                sum_variance = sum_variance + ((avg_price - minutes[element_index][1]) ** 2)
            variance = sum_variance / timeseries_length
            stddev_price = math.sqrt(variance)

            if stddev_price == 0:
                minute_index = minute_index + skip_length
                continue

            for element_index in range(minute_index - timeseries_length_minus_one, minute_index + 1):
                window_minutes.append(np.array([(minutes[element_index][1] - avg_price) / stddev_price,
                                                minutes[element_index][2],
                                                minutes[element_index][3],
                                                minutes[element_index][4],
                                                minutes[element_index][5],
                                                minutes[element_index][6]
                                            ], dtype=np.float32))

            session_data.append(np.array(window_minutes, dtype=np.float32))
            session_labels.append(minutes[minute_index][7] + 1)
            minute_index = minute_index + skip_length

        data.append(np.array(session_data))
        labels.append(np.array(session_labels))

        memory_info = process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(f"load_data() - start_date: {session[0]}, count: {len(labels)}, memory: {array_memory:,}")

    return np.array(data, dtype=np.float32), np.array(labels)




def get_model(layer_count, base_units, flatten_units, dropout_percent, recurrent_dropout_percent, weight_decay):
    if layer_count <= 0:
        print(f"invalid layer_count: {layer_count}")
        exit(1)

    if flatten_units > 0:
        lstm_layer_count = layer_count
    else:
        lstm_layer_count = layer_count + 1

    dropout = dropout_percent / 100.0
    recurrent_dropout = recurrent_dropout_percent / 100.0
    weight_decay_fraction = weight_decay / 10000.0

    model = tf.keras.models.Sequential()
    print("tf.keras.models.Sequential([")

    for label_index in range(0, lstm_layer_count):
        layer_power = lstm_layer_count - label_index - 1
        layer_multiple = 1
        for i in range(layer_power):
            layer_multiple = layer_multiple * 2
        layer_units = base_units * layer_multiple

        if flatten_units > 0:
            return_sequences = True
        elif layer_power > 0:
            return_sequences = True
        else:
            return_sequences = False

        if weight_decay > 0:
            model.add(tf.keras.layers.LSTM(layer_units,
                                           dropout=dropout,
                                           recurrent_dropout=recurrent_dropout,
                                           return_sequences=return_sequences,
                                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay_fraction),
                                           recurrent_regularizer=tf.keras.regularizers.l2(weight_decay_fraction)))
            print(f"\ttf.keras.layers.LSTM({layer_units},")
            print(f"\t\tdropout={dropout},")
            print(f"\t\trecurrent_dropout={recurrent_dropout},")
            print(f"\t\treturn_sequences={return_sequences},")
            print(f"\t\tkernel_regularizer=tf.keras.regularizers.l2({weight_decay_fraction}),")
            print(f"\t\trecurrent_regularizer=tf.keras.regularizers.l2({weight_decay_fraction})),")

        else:
            model.add(tf.keras.layers.LSTM(layer_units,
                                           dropout=dropout,
                                           recurrent_dropout=recurrent_dropout,
                                           return_sequences=return_sequences))
            print(f"\ttf.keras.layers.LSTM({layer_units},")
            print(f"\t\tdropout={dropout},")
            print(f"\t\trecurrent_dropout={recurrent_dropout},")
            print(f"\t\treturn_sequences={return_sequences}),")

    if flatten_units > 0:
        model.add(tf.keras.layers.Flatten())
        print(f"\ttf.keras.layers.Flatten(),")

        if weight_decay > 0:
            model.add(tf.keras.layers.Dense(flatten_units,
                                            activation="relu",
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay_fraction)))
            print(f"\ttf.keras.layers.Dense({flatten_units},")
            print(f"\t\tactivation='relu',")
            print(f"\t\tkernel_regularizer=tf.keras.regularizers.l2({weight_decay_fraction}))")
        else:
            model.add(tf.keras.layers.Dense(flatten_units,
                                            activation="relu"))
            print(f"\ttf.keras.layers.Dense({flatten_units},")
            print(f"\t\tactivation='relu')")

        model.add(tf.keras.layers.Dropout(0.2))
        print(f"\ttf.keras.layers.Dropout(0.2)")

    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    print(f"\ttf.keras.layers.Dense(3, activation='softmax')]")

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
