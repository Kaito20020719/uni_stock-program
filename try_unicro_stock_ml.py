import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

ticker = "9983.T"
data = yf.download(ticker , period = "20y",interval="1d")

# RSIの計算関数
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

#移動平均の追加
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA25"] = data["Close"].rolling(window=25).mean()
data["RSI"] = calculate_rsi(data["Close"])

# 変化率（前日比）の計算
data["Return"] = data["Close"].pct_change()

# 出来高も追加（急騰の予兆を掴むため）
data["Vol_Change"] = data["Volume"].pct_change()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

raw_data = data[["Return","Close", "MA5", "MA25", "RSI", "Vol_Change"]].dropna().values
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(raw_data)


#print(type(raw_data))   
#print(f"正規化前の最初の5行:\n{raw_data[:5]}")
#print(f"正規化後の最初の5行:\n{scaled_data[:5]}")

look_back = 30
X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back])
    y.append(scaled_data[i + look_back, 0])  # Close価格を予測対象とする

X = np.array(X)
y = np.array(y).reshape(-1, 1)

#データの分割
train_size = int(len(X) * 0.8)

test_dates = data.index[look_back + train_size:]

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

#print(f"全データ数:{len(X)}")
#print(f"学習データ数:{len(X_train)}")
#print(f"テストデータ数:{len(X_test)}")

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

class StockPredictor(nn.Module):
    def __init__(self,input_dim,hiddden_dim,num_layers,output_dim):
        super(StockPredictor,self).__init__()
        self.hiddden_dim = hiddden_dim
        self.num_layers = num_layers


        #LSTM層
        self.lstm = nn.LSTM(input_dim,hiddden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hiddden_dim,output_dim) #出力層
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hiddden_dim)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hiddden_dim)

        #LSTM層を通過
        out,(hn,cn) = self.lstm(x)

        #最後のタイムステップの出力を使って次の日の株価を予測
        out = self.fc(out[:,-1,:]) #最後のタイムステップの出力を使用
        return out

input_dim = 6
hidden_dim = 64
num_layers = 3
output_dim = 1

#インスタンス化
model = StockPredictor(input_dim,hidden_dim,num_layers,output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

num_epochs = 500
train_hist = np.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    train_hist[epoch] = loss.item()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# plt.figure(figsize=(8, 4))
# plt.plot(train_hist, label='Training Loss')
# plt.title("Loss History")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.show()

model.eval()

with torch.no_grad():
    pred_returns_scaled = model(X_test).numpy()

# 変化率として逆変換
dummy = np.zeros((len(pred_returns_scaled), 6))
dummy[:, 0] = pred_returns_scaled.flatten()
pred_returns = scaler.inverse_transform(dummy)[:, 0]

def get_inverse_transform(scaler, scaled_val):
    dummy = np.zeros((len(scaled_val),6))
    dummy[:,0] = scaled_val.flatten()  # Close価格を予測対象とする
    return scaler.inverse_transform(dummy)[:,0]  # Close価格の列を返す

# 実際の株価(Close)に変換
# テスト期間の「前日の終値」を取得
actual_close_start = data["Close"].iloc[train_size + look_back - 1 : -1].values
predicted_close = actual_close_start * (1 + pred_returns)
actual_close = data["Close"].iloc[train_size + look_back:].values

# plt.figure(figsize=(12, 6))
# plt.plot(testpredicted_p, label='Test Prediction')
# plt.plot(testy_p, label='Test Actual')
# plt.title("Test Prediction vs Actual")
# plt.xlabel("Time")
# plt.ylabel("Stock Price")
# plt.legend()
# plt.grid()
# plt.show()

num_test_sample = len(y_test)
test_dates = data.index[-num_test_sample:]
# print(f"日付の数: {len(test_dates)}")
# print(f"予測データの数: {len(testy_p)}")

plt.figure(figsize=(12, 6)) 

# 横軸に test_dates を指定
plt.plot(test_dates, actual_close, label='Actual Price', color='blue')
plt.plot(test_dates, predicted_close, label='AI Prediction', color='red', linestyle='--')

plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price (JPY)")
plt.legend()
plt.grid(True)

# 日付ラベルを斜めにして見やすくする
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()