"""Predictions.ipynb
لینک در کولب
    https://colab.research.google.com/drive/1baCVlxHOT3xCuOXqWo_wED7AIESylQ67

وبسایت یاهو فایننس
    https://finance.yahoo.com
برای استفاده از کتابخانه tensorflow
باید از پایتون 3.11.7 max استفاده کرد
"""
#نصب کتابخانه ها
pip install numpy

pip install matplotlib

pip install pandas-datareader

pip install scikit-learn

pip install tensorflow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow


from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance

# لود کردن دیتا
# اسم کمپانی سهام
company = 'META'

#تاریخ شروع و پایان برای ایجاد دیتاست
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

#دریافت دیتاست از سایت یاهو فایننس
data = yfinance.download(company, start, end)

#آماده سازی داده
#تبدیل قیمت های بسته شده به اعدادی بین 0 و 1 
scaler = MinMaxScaler(feature_range=(0,10))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#دریافت 60 روز داده از قبل
prediction_days = 60

#لیسا خالی که بعدا با دادهای تمرین پر میشود
x_train = []
y_train = []

# از 60 تا اخرین ایندکس
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

#تبدیل به آرایه نامپای
x_train, y_train = np.array(x_train), np.array(y_train)

# ریشبپ کردن برای اینکه با شبکه عصبی کار کند
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#ساختن مدل
#مدل ساده شبکه عصبی
model = Sequential()


#تعریف لایه ها
#یونیت تعداد لایه ها, اگه لایه زیاد باشه اور فیت میشود
#ریتورن سکونسز قراره همواره از عقب داده بگیره
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

#لایه اخر لایه پیش بینی
model.add(Dense(units=1))

#مدل را تفیسر میکنیم با بهینه ساز آدام
model.compile(optimizer='adam', loss='mean_squared_error')

# مدل را فیت میکنیم
#قراره مدل داده را 24 بار ببینه
#قراره 32 لایه رو همزمان هم ببینه
model.fit(x_train,y_train, epochs=25, batch_size=32)

'''تست دقت مدل بر داده ای که وجود دارد'''

#آماده سازی و تهیه داده تست
#این داده باید داده ای باشدکه مدل قبلا ندیده است
test_start = dt.datetime(2020,1,1)
#تا الان
test_end = dt.datetime.now()

test_data = yfinance.download(company, test_start, test_end)
actual_prices = test_data['Close'].values

# داده های تمرین و تست  اینجا میشینن
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# ورودی مدل
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# پیش بینی بروی داده تست
x_test = []

#+1 یک روز هم از پیش بینی اصلی را رسم میکند
for x in range(prediction_days, len(model_inputs)+1):
    x_test.append(model_inputs[x-prediction_days:x, 0])

#تبدیل به ارایه نامپای
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#پیشبینی
predicted_prices = model.predict(x_test)

# تبدیل داده های اسکیل شده به داده معمولی
predicted_prices = scaler.inverse_transform(predicted_prices)

# رسم چارت پیشبینی تست
plt.plot(actual_prices, color="black" , label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

#دقت پیش بینی به درصد
accuracy = model.evaluate(actual_prices, predicted_prices)
print('Accuracy: %.2f' % (accuracy * 100))

#پیش بینی روز بعد
#ایجاد یک دیتاست جدید
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0], real_data.shape[1],1))

#اختیاری
print(scaler.inverse_transform(real_data[-1]))

#پیشبینی و تبدیل به داده بدرد بخور
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")