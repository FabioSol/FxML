from FxML.TimeFrames import TimeFrame



class Bar:
    def __init__(self, timeframe: TimeFrame, open_: float, high: float, low: float, close: float, volume: int = None):
        self.values = [TimeFrame(timeframe).value, open_, high, low, close, volume]
        self.timeframe = TimeFrame(timeframe).value
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __add__(self, other: 'Bar') -> 'Bar':
        if other is None:
            # If other is None, return a copy of self
            return self.copy()
        elif self is None:
            # If self is None, return a copy of other
            return other.copy()
        else:
            tf = self.timeframe + other.timeframe
            return Bar(timeframe=tf, open_=self.open, high=max(self.high, other.high),
                       low=min(self.low, other.low), close=other.close, volume=self.volume + other.volume)

    def __str__(self):
        return "Bar[{timeframe:<4}, {open_:>5}, {high:>5}, {low:>5}, {close:>5}, {volume:>5}]".format(
            timeframe=str(self.timeframe),
            open_=str(round(self.open, 4)),
            high=str(round(self.high, 4)),
            low=str(round(self.low, 4)),
            close=str(round(self.close, 4)),
            volume=str(self.volume)
        )

    def __repr__(self):
        return self.__str__()

    def range(self):
        return self.high - self.low

    def pct_range(self):
        return self.range() / self.low

    def pct_change(self):
        return self.close / self.open - 1

    def microbar(self):
        self.values = [self.pct_change(), self.pct_range(), self.volume]
        return self.values

    def no_process(self):
        return self

    @staticmethod
    def upper_threshold(threshold):
        def function(self):
            return 1 if self.pct_change() > threshold else 0

        return function

    def lower_threshold(self, threshold):
        return 1 if self.pct_change() < threshold else 0

