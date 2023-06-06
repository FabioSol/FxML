import numpy as np
import pandas as pd
from FxML.Bars import Bar
from FxML.TimeFrames import TimeFrame


class Hist(np.ndarray):
    def __new__(cls, input_array):
        if isinstance(input_array, pd.DataFrame):
            bars = Hist.df_to_hist(input_array)
        else:
            bars = input_array

        cls.previous = [1]
        cls.future = [1]

        cls.x_bar_process = lambda _, obj: obj
        cls.y_bar_process = lambda _, obj: obj

        cls.flatten = False
        cls.seed = 45136

        cls.len_override = False
        cls.len_override_value = 0

        cls.data = np.array(bars).view(cls)
        cls.values_dict = {num: num for num in range(0, len(cls.data) + 1)}
        return cls.data

    def __str__(self):
        return '\n'.join(Bar.__str__(bar) for bar in np.asarray(self).__iter__())

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def df_to_hist(df: pd.DataFrame, hastimeframe: bool = False, timeframe=TimeFrame("M1")):
        if hastimeframe:
            return Hist(
                list(map(lambda x: Bar(*x), df[["Timeframe", "Open", "High", "Low", "Close", "Tick volume"]].values)))
        else:
            return Hist(
                list(map(lambda x: Bar(timeframe, *x), df[["Open", "High", "Low", "Close", "Tick volume"]].values)))

    def set_view(self, previous: list, future: list):
        self.previous = previous
        self.future = future
        self.values_dict = {num: num for num in range(0, len(self))}

    def x_bar_processing(self, method=Bar.microbar, flatten=True):
        self.x_bar_process = method
        self.flatten = flatten

    def y_bar_processing(self, method=Bar.pct_change, flatten=True):
        self.y_bar_process = method
        self.flatten = flatten

    def __getitem__(self, index, x: bool = False, y: bool = False):
        if isinstance(index, int):
            index = self.values_dict[index]
            start = index
            end = sum(self.previous) + sum(self.future) + index + 1
            sizes_of_bars = self.previous + [1] + self.future

            bars = [None for _ in range(len(sizes_of_bars))]
            limits = np.cumsum(sizes_of_bars) + index
            current_bar_num = 0
            bars[0] = super().__getitem__(start)
            for i in range(start + 1, end):
                if i >= limits[current_bar_num]:
                    current_bar_num += 1
                    bars[current_bar_num] = super().__getitem__(i)
                else:
                    bars[current_bar_num] += super().__getitem__(i)

            if x:
                bars = bars[:len(self.previous) + 1]
                bars = [self.x_bar_process(bar) for bar in bars]
                if self.flatten & isinstance(bars[0], list):
                    bars = [element for sublist in bars for element in sublist]
            elif y:
                bars = bars[len(self.previous) + 1:]
                bars = [self.y_bar_process(bar) for bar in bars]
                if self.flatten & isinstance(bars[0], list):
                    bars = [element for sublist in bars for element in sublist]

            return np.array(bars)

        elif isinstance(index, slice):

            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self)
            step = index.step if index.step is not None else 1

            bars = []
            for i in range(start, stop, step):
                bars += [self.__getitem__(i)]

            return np.array(bars)

    def __len__(self):
        if self.len_override:
            return self.len_override_value
        else:
            return super().__len__() - sum(self.future) - sum(self.previous)

    def x(self):
        x = _X(self)
        x.previous = self.previous
        x.future = self.future
        x.x_bar_process = self.x_bar_process
        x.flatten = self.flatten
        x.values_dict = {num: num for num in range(0, len(x))}
        return x

    def y(self):
        y = _Y(self)
        y.previous = self.previous
        y.future = self.future
        y.y_bar_process = self.y_bar_process
        y.flatten = self.flatten
        y.values_dict = {num: num for num in range(0, len(y))}
        return y

    def pct_candles(self):
        self.data = [bar.microbar() for bar in np.asarray(self.data)]
        return self.data

    def set_seed(self, seed):
        self.seed = seed

    def random_partition(self, test_size: float = 0.2):
        n = len(self)
        test_len = round(n * test_size)
        train_len = n - test_len
        train_keys = np.arange(train_len)
        test_keys = np.arange(test_len)

        randomized_array = np.random.permutation(n)
        randomized_array_train = randomized_array[:train_len]
        randomized_array_test = randomized_array[:test_len]

        train_dict = dict(zip(train_keys, randomized_array_train))
        test_dict = dict(zip(test_keys, randomized_array_test))

        X_train, X_test, y_train, y_test = _X(self), _X(self), _Y(self), _Y(self)
        X_train.previous, X_test.previous, y_train.previous, y_test.previous = self.previous, self.previous, self.previous, self.previous
        X_train.future, X_test.future, y_train.future, y_test.future = self.future, self.future, self.future, self.future
        X_train.x_bar_process, X_test.x_bar_process, y_train.y_bar_process, y_test.y_bar_process = self.x_bar_process, self.x_bar_process, self.y_bar_process, self.y_bar_process
        X_train.flatten, X_test.flatten, y_train.flatten, y_test.flatten = self.flatten, self.flatten, self.flatten, self.flatten
        X_train.len_override, X_test.len_override, y_train.len_override, y_test.len_override = True, True, True, True
        X_train.len_override_value, X_test.len_override_value, y_train.len_override_value, y_test.len_override_value = train_len, test_len, train_len, test_len
        X_train.values_dict, X_test.values_dict, y_train.values_dict, y_test.values_dict = train_dict, test_dict, train_dict, test_dict
        return X_train, X_test, y_train, y_test


class _X(Hist):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return super().__getitem__(index, x=True, y=False)


class _Y(Hist):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return super().__getitem__(index, x=False, y=True)


class Dim1Hist(np.ndarray):
    def __new__(cls, bars, loc):
        cls.location = loc
        arr = np.array(bars, dtype=object)
        obj = np.asarray(arr).view(cls)
        return obj

    def __str__(self):
        parent_str = super().__str__()
        counter = 0
        start = -1
        end = -1

        for i in range(len(parent_str)):
            if parent_str[i] == ' ':
                counter += 1
            if counter == 5 * self.location:
                if start == -1:
                    start = i
            if counter == 5 * (self.location + 1):
                if end == -1:
                    end = i

        return parent_str[:start + 1] + "{" + parent_str[start:end + 1] + "}" + parent_str[end:]

    def __repr__(self):
        return "(" + self.__str__() + ")"
