import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("ccpp.csv")

df.rename(columns={'AT': 'Average Temperature', 'V': 'Exhaust Vacuum','AP': 'Ambient Pressure',
                   'RH': 'Relative Humidity ','PE': 'Net Hourly Electrical Energy Output'}, inplace=True)

predictors = df.drop("Net Hourly Electrical Energy Output", axis=1).values
targets = df["Net Hourly Electrical Energy Output"].values

X_train, X_test, y_train, y_test = train_test_split(predictors,targets, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def get_new_model():
  model = Sequential()
  model.add(Dense(6, activation='relu'))
  model.add(Dense(6, activation='relu'))
  model.add(Dense(1))

  return model

model = get_new_model()
model.compile(optimizer = 'adam', loss="mean_squared_error")

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=4)

history = model.fit(x=X_train, y=y_train, epochs=100, batch_size=32, validation_data=(X_test,y_test), callbacks=[early_stopping_monitor])

plt.style.use("ggplot")
pd.DataFrame(model.history.history).plot(figsize=(12,10))
