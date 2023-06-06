import pandas as pd
from FxML import Bars
from FxML.Hists import Hist

from sklearn.linear_model import SGDRegressor

model = SGDRegressor()

h = Hist.df_to_hist(pd.read_csv("../data/Example_data.csv").head(30))
prev_candles = [5, 2, 2]
future_candles = [2]
h.set_view(prev_candles, future_candles)
h.x_bar_processing()
h.y_bar_processing(Bars.Bar.upper_threshold(0.000004))
X_train, X_test, y_train, y_test = h.random_partition()

for i in range(len(X_train)):
    # Extract the current row
    X_row = X_train[i].reshape(1, -1)
    y_row = y_train[i].ravel()

    # Fit the model on the current row
    model.partial_fit(X_row, y_row)
    prediction = model.predict(X_row)
    print("Input:", X_row)
    print("Target:", y_row)
    print("Prediction:", prediction)
    print(i)
