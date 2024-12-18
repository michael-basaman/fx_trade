import numpy as np
import tensorflow as tf

import datetime
import hashlib
import math
import os
import psutil
import psycopg2
import shutil
import time

from sklearn.model_selection import train_test_split


class EvaluateTestDataCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, test_labels, checkpoint_path):
        self._test_data = test_data
        self._test_labels = test_labels
        self._checkpoint_path = checkpoint_path
        self._md5sum = ""
        self._loss = 0
        self._accuracy = 0
        self._last_time = 0

    def get_md5sum(self):
        with open(self._checkpoint_path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

        return file_hash.hexdigest()

    def on_epoch_begin(self, epoch, logs=None):
        now = time.time()

        if epoch == 0:
            self._last_time = now
            return

        print(f"Epoch {epoch}: ran in {format_seconds(now - self._last_time)}")
        self._last_time = now

        if os.path.isfile(self._checkpoint_path):
            md5sum = self.get_md5sum()
            if md5sum != self._md5sum:
                loss, accuracy = self.model.evaluate(self._test_data, self._test_labels, verbose=1)

                self._md5sum = md5sum
                self._loss = loss
                self._accuracy = accuracy

                print(f"Epoch {epoch}: new checkpoint: {md5sum}, test accuracy: {self._accuracy:.6f}, loss: {self._accuracy:.6f}")
            else:
                print(f"Epoch {epoch}: same checkpoint: {md5sum}, test accuracy: {self._accuracy:.6f}, loss: {self._accuracy:.6f}")
        else:
            print(f"Epoch {epoch}: checkpoint {self._checkpoint_path} not found")


class FxTrainer():
    def __init__(self):
        self._process = psutil.Process()
        self._now = datetime.datetime.now()
        self._now_str = self._now.strftime("%Y%m%d%H%M%S")
        self._seed_str = self._now.strftime("%m%d%H%M%S")
        self._seed = int(self._seed_str)

    def run(self):
        timeseries_length = 30
        skip_length = 1
        outcome_minutes = 15
        pips = 850
        patience = 10
        self_split = True

        start_time = time.time()

        memory_info = self._process.memory_info()
        initial_memory = memory_info.rss

        (train_data, train_labels,
         val_data, val_labels,
         test_data, test_labels,
         class_weight) = self.get_data(timeseries_length, skip_length, outcome_minutes, pips, self_split)

        memory_info = self._process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(
            f"Loaded {len(train_data) + len(val_data) + len(test_data)} minutes in {format_seconds(time.time() - start_time)}, memory: {array_memory:,}")

        dropouts = [(0, 0, 0), (20, 0, 0), (20, 10, 0), (20, 10, 1), (20, 10, 10)]

        for layer_count in [3, 2, 4]:
            for base_units in [128, 64, 256]:
                for flatten in [False, True]:
                    for dropout_percent, recurrent_dropout_percent, weight_decay in dropouts:
                        start_time = time.time()

                        checkpoint_path = f"C:/VirtualBox/rsync/fx_trade/checkpoints/{self._now_str}_c{layer_count}_u{base_units:03d}_f{'T' if flatten else 'F'}_d{dropout_percent:02d}_r{recurrent_dropout_percent:02d}_w{weight_decay:02d}.model.keras"

                        tf.keras.backend.clear_session()

                        model = self.get_model(layer_count,
                                               base_units,
                                               flatten,
                                               dropout_percent,
                                               recurrent_dropout_percent,
                                               weight_decay)

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

                        evaluate_test_data = EvaluateTestDataCallback(test_data, test_labels, checkpoint_path)

                        model.fit(train_data, train_labels,
                                  epochs=10000,
                                  callbacks=[early_stopping, cp_callback, evaluate_test_data],
                                  # class_weight=class_weight,
                                  validation_data=(val_data, val_labels))

                        print(f"trained model in {format_seconds(time.time() - start_time)}")
                        start_time = time.time()

                        loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)

                        print(f"accuracy: {accuracy:.6f}, loss: {loss:.6f}")

                        model_path = f"C:/VirtualBox/rsync/fx_trade/models/{self._now_str}_l{int(10000 * loss):05d}_a{int(10000 * accuracy):05d}_c{layer_count}_u{base_units:03d}_f{'T' if flatten else 'F'}_d{dropout_percent:02d}_r{recurrent_dropout_percent:02d}_w{weight_decay:02d}.model.keras"
                        shutil.move(checkpoint_path, model_path)

        print(f"finished in {format_seconds(time.time() - start_time)}")

    def get_data(self, timeseries_length, skip_length, outcome_minutes, pips, self_split):
        memory_info = self._process.memory_info()
        initial_memory = memory_info.rss

        if self_split:
            (train_data_840, train_data_600, train_labels_840, train_labels_600,
             val_data_840, val_data_600, val_labels_840, val_labels_600,
             test_data_840, test_data_600, test_labels_840, test_labels_600,
             class_weight) = self.get_split(timeseries_length, skip_length, outcome_minutes, pips)

            train_data = np.concatenate((train_data_840, train_data_600), axis=0)
            train_labels = np.concatenate((train_labels_840, train_labels_600), axis=0)
            val_data = np.concatenate((val_data_840, val_data_600), axis=0)
            val_labels = np.concatenate((val_labels_840, val_labels_600), axis=0)
            test_data = np.concatenate((test_data_840, test_data_600), axis=0)
            test_labels = np.concatenate((test_labels_840, test_labels_600), axis=0)

            indices = np.random.default_rng(seed=self._seed).permutation(len(train_data))
            self._seed = self._seed + 1

            train_data = train_data[indices]
            train_labels = train_labels[indices]

            indices = np.random.default_rng(seed=self._seed).permutation(len(val_data))
            self._seed = self._seed + 1

            val_data = val_data[indices]
            val_labels = val_labels[indices]

            indices = np.random.default_rng(seed=self._seed).permutation(len(test_data))
            self._seed = self._seed + 1

            test_data = test_data[indices]
            test_labels = test_labels[indices]
        else:
            data, labels = self.load_data(timeseries_length, skip_length, outcome_minutes, pips, False)

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

            nontest_data, test_data, nontest_labels, test_labels = train_test_split(
                data, labels, test_size=0.2
            )

            train_data, val_data, train_labels, val_labels = train_test_split(
                nontest_data, nontest_labels, test_size=0.2
            )

            print(f"class_weight: {class_weight}")

        memory_info = self._process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(
            f"get_data() - train_data: {len(train_data)}, val_data: {len(val_data)}, test_data: {len(test_data)}, memory: {array_memory:,}")

        return (train_data, train_labels,
                val_data, val_labels,
                test_data, test_labels,
                class_weight)


    def get_split(self, timeseries_length, skip_length, outcome_minutes, pips):
        memory_info = self._process.memory_info()
        initial_memory = memory_info.rss

        data_840, labels_840, data_600, labels_600 = self.load_data(timeseries_length, skip_length, outcome_minutes, pips, True)

        memory_info = self._process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(f"get_split() - data_840: {len(data_840)}, data_600: {len(data_600)}, memory: {array_memory:,}")

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

        indices = np.random.default_rng(seed=self._seed).permutation(len(data_840))
        self._seed = self._seed + 1

        data_840 = data_840[indices]
        labels_840 = labels_840[indices]

        indices = np.random.default_rng(seed=self._seed).permutation(len(data_600))
        self._seed = self._seed + 1

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

        train_data_840 = np.reshape(train_data_840, (train_data_840.shape[0] * train_data_840.shape[1], train_data_840.shape[2], train_data_840.shape[3]))
        train_labels_840 = np.reshape(train_labels_840, (train_labels_840.shape[0] * train_labels_840.shape[1]))
        val_data_840 = np.reshape(val_data_840, (val_data_840.shape[0] * val_data_840.shape[1], val_data_840.shape[2], val_data_840.shape[3]))
        val_labels_840 = np.reshape(val_labels_840, (val_labels_840.shape[0] * val_labels_840.shape[1]))
        test_data_840 = np.reshape(test_data_840, (test_data_840.shape[0] * test_data_840.shape[1], test_data_840.shape[2], test_data_840.shape[3]))
        test_labels_840 = np.reshape(test_labels_840, (test_labels_840.shape[0] * test_labels_840.shape[1]))

        train_data_600 = np.reshape(train_data_600, (train_data_600.shape[0] * train_data_600.shape[1], train_data_600.shape[2], train_data_600.shape[3]))
        train_labels_600 = np.reshape(train_labels_600, (train_labels_600.shape[0] * train_labels_600.shape[1]))
        val_data_600 = np.reshape(val_data_600, (val_data_600.shape[0] * val_data_600.shape[1], val_data_600.shape[2], val_data_600.shape[3]))
        val_labels_600 = np.reshape(val_labels_600, (val_labels_600.shape[0] * val_labels_600.shape[1]))
        test_data_600 = np.reshape(test_data_600, (test_data_600.shape[0] * test_data_600.shape[1], test_data_600.shape[2], test_data_600.shape[3]))
        test_labels_600 = np.reshape(test_labels_600, (test_labels_600.shape[0] * test_labels_600.shape[1]))

        memory_info = self._process.memory_info()
        array_memory = memory_info.rss - initial_memory
        print(f"get_split() - train_data_840: {len(train_data_840)}, train_data_600: {len(train_data_600)}, memory: {array_memory:,}")

        return (train_data_840, train_data_600, train_labels_840, train_labels_600,
                val_data_840, val_data_600, val_labels_840, val_labels_600,
                test_data_840, test_data_600, test_labels_840, test_labels_600,
                class_weight)

    def load_data(self, timeseries_length, skip_length, outcome_minutes, pips, self_split):
        conn = psycopg2.connect(database="fx", user="fx", password="fx", host="localhost", port=5432)

        cursor = conn.cursor()
        cursor2 = conn.cursor()
        cursor3 = conn.cursor()

        normals_d = {}
        cursor3.execute("SELECT name, value FROM normals")

        normals = cursor3.fetchall()

        for normal in normals:
            normals_d[normal[0]] = normal[1]

        cursor.execute("""
        SELECT start_time, end_time
        FROM sessions
        WHERE complete is true
        AND holiday is false
        order by start_time
        """)

        sessions = cursor.fetchall()

        memory_info = self._process.memory_info()
        initial_memory = memory_info.rss

        data_840 = []
        labels_840 = []
        data_600 = []
        labels_600 = []
        data = []
        labels = []

        outcome_seconds = outcome_minutes * 60

        for session in sessions:
            cursor2.execute("""
            SELECT m.fx_datetime,
                   m.close_price,
                   m.close_price - m.min_price above,
                   m.close_price - m.open_price candle,
                   m.max_price - m.min_price tail,
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

                for element_index in range(minute_index - timeseries_length_minus_one, minute_index + 1):
                    window_minutes.append(np.array([(minutes[element_index][1] - avg_price) / normals_d[f"sma_{timeseries_length}_stddev"],
                                                    (minutes[element_index][2] - normals_d["above_average"]) / normals_d["above_stddev"],
                                                    (minutes[element_index][3] - normals_d["candle_average"]) / normals_d["candle_stddev"],
                                                    (minutes[element_index][4] - normals_d["tail_average"]) / normals_d["tail_stddev"],
                                                ], dtype=np.float32))

                if self_split:
                    session_data.append(np.array(window_minutes, dtype=np.float32))
                    session_labels.append(minutes[minute_index][5] + 1)
                else:
                    data.append(np.array(window_minutes, dtype=np.float32))
                    labels.append(minutes[minute_index][5] + 1)

                minute_index = minute_index + skip_length

            if self_split:
                if len(minutes) == 840:
                    data_840.append(np.array(session_data))
                    labels_840.append(np.array(session_labels))
                elif len(minutes) == 600:
                    data_600.append(np.array(session_data))
                    labels_600.append(np.array(session_labels))

            memory_info = self._process.memory_info()
            array_memory = memory_info.rss - initial_memory
            print(f"load_data() - start_date: {session[0]}, count: {len(data_840) + len(data_600)}, memory: {array_memory:,}")

        if self_split:
            return np.array(data_840, dtype=np.float32), np.array(labels_840), np.array(data_600, dtype=np.float32), np.array(labels_600)
        else:
            return np.array(data, dtype=np.float32), np.array(labels)

    def get_model(self, layer_count, base_units, flatten, dropout_percent, recurrent_dropout_percent, weight_decay, print_model=True):
        if layer_count <= 0:
            print(f"invalid layer_count: {layer_count}")
            exit(1)

        dropout = dropout_percent / 100.0
        recurrent_dropout = recurrent_dropout_percent / 100.0
        weight_decay_fraction = weight_decay / 10000.0

        model = tf.keras.models.Sequential()

        if print_model:
            print(f"get_model({layer_count}, {base_units}, {flatten}, {dropout_percent}, {recurrent_dropout_percent}, {weight_decay})")

        for label_index in range(0, layer_count):
            layer_power = layer_count - label_index - 1
            layer_multiple = 1
            for i in range(layer_power):
                layer_multiple = layer_multiple * 2
            layer_units = base_units * layer_multiple

            if flatten:
                return_sequences = True
            elif layer_power > 0:
                return_sequences = True
            else:
                return_sequences = False

            if weight_decay > 0:
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_units,
                                               dropout=dropout,
                                               recurrent_dropout=recurrent_dropout,
                                               return_sequences=return_sequences,
                                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay_fraction),
                                               recurrent_regularizer=tf.keras.regularizers.l2(weight_decay_fraction))))

                if print_model:
                    if label_index == 0:
                        print("Sequential([")

                    print(f"    Bidirectional(LSTM({layer_units},")
                    print(f"                       dropout={dropout},")
                    print(f"                       recurrent_dropout={recurrent_dropout},")
                    print(f"                       return_sequences={return_sequences},")
                    print(f"                       kernel_regularizer=l2({weight_decay_fraction}),")
                    print(f"                       recurrent_regularizer=l2({weight_decay_fraction})),")

            else:
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_units,
                                               dropout=dropout,
                                               recurrent_dropout=recurrent_dropout,
                                               return_sequences=return_sequences)))

                if print_model:
                    if label_index == 0:
                        print("Sequential([")

                    print(f"    Bidirectional(LSTM({layer_units},")
                    print(f"                       dropout={dropout},")
                    print(f"                       recurrent_dropout={recurrent_dropout},")
                    print(f"                       return_sequences={return_sequences}),")

        if flatten:
            model.add(tf.keras.layers.Flatten())

            if print_model:
                print(f"    Flatten(),")

        if weight_decay > 0:
            model.add(tf.keras.layers.Dense(base_units,
                                            activation="relu",
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay_fraction)))

            if print_model:
                print(f"    Dense({base_units},")
                print(f"          activation='relu',")
                print(f"          kernel_regularizer=tf.keras.regularizers.l2({weight_decay_fraction})),")
        else:
            model.add(tf.keras.layers.Dense(base_units, activation="relu"))

            if print_model:
                print(f"    Dense({base_units}, activation='relu'),")

        model.add(tf.keras.layers.Dropout(0.2))

        if print_model:
            print(f"    Dropout(0.2),")

        model.add(tf.keras.layers.Dense(3, activation="softmax"))

        if print_model:
            print(f"    Dense(3, activation='softmax')")
            print(f"]")

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


if __name__ == "__main__":
    trainer = FxTrainer()
    trainer.run()
