class TimeFrame:
    def __init__(self, timeframe):
        if isinstance(timeframe, TimeFrame):
            self.string = timeframe.string
            self.value = timeframe.value

        elif isinstance(timeframe, int):
            if timeframe % 10080 == 0:
                t = "W"
                n = timeframe // 10080
            elif timeframe % 1440 == 0:
                t = "D"
                n = timeframe // 1440
            elif timeframe % 60 == 0:
                t = "H"
                n = timeframe // 60
            else:
                t = "M"
                n = timeframe

            self.string = t + str(int(n))
            self.value = timeframe
        else:
            self.string = timeframe
            numbers = ''.join(filter(str.isdigit, timeframe))
            letters = ''.join(filter(str.isalpha, timeframe))

            multipliers = {'M': 1, 'H': 60, 'D': 1440, 'W': 10080}

            minutes = int(numbers) * multipliers.get(letters, 1)
            self.value = minutes

    def __str__(self):
        return self.string

    def __add__(self, other):
        if isinstance(other, TimeFrame):
            return TimeFrame(int(self.value) + int(other.value))
        elif isinstance(other, int):
            return TimeFrame(int(self.value) + other)
        else:
            return NotImplemented


