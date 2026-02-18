import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from pygooglenews import GoogleNews
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re
import google.generativeai as genai
from google.generativeai import types



###特徴量の準備
ticker = "9983.T"
data = yf.download(ticker , period = "20y",interval="1d")

data = pd.DataFrame(data.values, 
                    index=data.index, 
                    columns=data.columns.get_level_values(0))

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

###ニュースデータ
def get_stock_news(ticker_name, lang='ja', country='JP'):
    gn = GoogleNews(lang=lang, country=country)
    # 検索ワードを指定（例：「ファーストリテイリング」）
    search = gn.search(ticker_name)
    
    news_list = []
    for entry in search['entries']:
        news_list.append({
            'date': entry.published,
            'title': entry.title
        })
    
    return pd.DataFrame(news_list)

# 実行
news_df = get_stock_news("ファーストリテイリング")
print(news_df.head())

def analyze_sentiment(text):
    pos_words = ['上昇', '好調', '増加', '成長', '成功','利益','好材料','買い','最高益','増収','増益',"支援","好業績","好決算","好調","好材料","好ニュース","好評","好感","好意的","期待",'上方修正','買いたい']
    neg_words = ['下落', '不調', '減少', '衰退', '失敗','損失','悪材料','売り','最悪','減収','減益',"懸念","悪業績","悪決算","不調","悪材料","悪ニュース","悪評","悪感","否定的","失望",'下方修正','売りたい','赤字','売り','衰退','厳しい']

    score = 0
    for word in pos_words:
        if word in text:
            score +=0.5
    for word in neg_words:
        if word in text:
            score -=0.5
    
    return max(min(score, 1), -1)

news_df['sentiment'] = news_df['title'].apply(analyze_sentiment)
news_df['date'] = pd.to_datetime(news_df['date']).dt.date
daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()

# APIキーを設定
genai.configure(api_key="AIzaSyDMZD72J9t0cbXiIrY4eigJDLRMlIbiSfk")
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

def analyze_sentiment_with_gemini(text):
    prompt = f"""
    以下の株価に関するニュースの見出しを読み、その内容がこの企業の株価にとって
    「ポジティブ」か「ネガティブ」かを判断して、-1.0から1.0の間の数値だけで答えてください。
    -1.0に近いほど超ネガティブ、1.0に近いほど超ポジティブ、0は中立です。
    余計な説明は一切不要です。数値のみを出力してください。

    ニュース: {text}
    """
    try:
        response = model_gemini.generate_content(prompt)
        # 数値だけを取り出す（念のため余計な文字を削除）
        score = float(response.text.strip())
        return max(min(score, 1.0), -1.0)
    except:
        # エラー（制限など）が発生した場合は中立(0)を返す
        return 0.0

# 適用する行を書き換え
#print(daily_sentiment.head())
# --- 修正：Geminiの分析直後に集計を行う ---
print("Geminiでニュースを分析中...")
news_df['sentiment'] = news_df['title'].apply(analyze_sentiment_with_gemini)

# 日付型に変換して集計（ここをGeminiの処理の後に持ってくる）
news_df['date'] = pd.to_datetime(news_df['date']).dt.date
daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
print("分析と集計が完了しました。")

data_reset = data.reset_index()
data_reset['Date'] = data_reset['Date'].dt.date

final_df = pd.merge(data_reset, daily_sentiment, left_on='Date', right_on='date', how='left')
final_df['sentiment'] = final_df['sentiment'].fillna(0)  # ニュースがない日は中立とみなす

feature_cols = ["Return","Close", "MA5", "MA25", "RSI", "Vol_Change","sentiment"]
final_data_cleaned = final_df.dropna(subset = feature_cols)

raw_data = final_data_cleaned[feature_cols].values
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(raw_data)


#print(type(raw_data))   
#print(f"正規化前の最初の5行:\n{raw_data[:5]}")
#print(f"正規化後の最初の5行:\n{scaled_data[:5]}")

###データセットの作成
look_back = 30
X, y = [], []
for i in range(len(scaled_data) - look_back - 1):
    X.append(scaled_data[i:i + look_back])
    target_return = final_data_cleaned["Return"].iloc[look_back + i + 1] # 予測対象は変化率（Return）
    y.append(1 if target_return > 0 else 0)  # 変化率を予測対象とする

X = np.array(X)
y = np.array(y).reshape(-1, 1)

#データの分割
train_size = int(len(X) * 0.8)

# y_testの数に合わせて後ろから日付を取る（一番安全な方法）
X_test = X[train_size:]
y_test = y[train_size:]
test_dates = final_data_cleaned["Date"].iloc[-len(y_test):]

X_train = X[:train_size]
y_train = y[:train_size]


#print(f"全データ数:{len(X)}")
#print(f"学習データ数:{len(X_train)}")
#print(f"テストデータ数:{len(X_test)}")

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

class StockClassifier(nn.Module):
    def __init__(self,input_dim,hiddden_dim,num_layers):
        super(StockClassifier,self).__init__()

        self.hiddden_dim = hiddden_dim
        self.num_layers = num_layers


        #LSTM層
        self.lstm = nn.LSTM(input_dim,hiddden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hiddden_dim,1) #出力層
        self.sigmoid = nn.Sigmoid() #シグモイド関数
    
    def forward(self,x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

input_dim = 7
hidden_dim = 64
num_layers = 3
output_dim = 1

#インスタンス化
model = StockClassifier(input_dim,hidden_dim,num_layers)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

num_epochs = 600
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
    y_prob = model(X_test).numpy() # 上がる確率
    y_pred = (y_prob > 0.5).astype(int) # 0.5以上なら「上がる」と判定

# 的中率の計算
accuracy = (y_pred == y_test.numpy()).mean()
print(f"テストデータの方向的中率: {accuracy*100:.2f}%")


# plt.figure(figsize=(12, 6))
# plt.plot(testpredicted_p, label='Test Prediction')
# plt.plot(testy_p, label='Test Actual')
# plt.title("Test Prediction vs Actual")
# plt.xlabel("Time")
# plt.ylabel("Stock Price")
# plt.legend()
# plt.grid()
# plt.show()

# num_test_sample = len(y_test)
# test_dates = data.index[-num_test_sample:]
# # print(f"日付の数: {len(test_dates)}")
# # print(f"予測データの数: {len(testy_p)}")

# plt.figure(figsize=(12, 6)) 

# # 横軸に test_dates を指定
# plt.plot(test_dates, actual_close, label='Actual Price', color='blue')
# plt.plot(test_dates, predicted_close, label='AI Prediction', color='red', linestyle='--')

# plt.title(f"{ticker} Stock Price Prediction")
# plt.xlabel("Date")
# plt.ylabel("Stock Price (JPY)")
# plt.legend()
# plt.grid(True)

# # 日付ラベルを斜めにして見やすくする
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# 可視化：直近50日間の「上昇確率」を表示
plt.figure(figsize=(12, 4))
plt.bar(range(50), y_prob[-50:].flatten(), color='orange', alpha=0.6, label='Up Probability')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.ylabel("Probability of Price Up")
plt.title("Prediction: Will the price go up tomorrow?")
plt.legend()
plt.show()




# --- 明日の予測用コード ---
model.eval()

# 1. 最新の30日分のデータを取得
# scaled_dataの最後から30行分が「今日までのデータ」
latest_data = scaled_data[-look_back:] 
latest_data_tensor = torch.from_numpy(latest_data).float().unsqueeze(0) # (1, 30, 7) の形に変換

with torch.no_grad():
    prediction_prob = model(latest_data_tensor).item()

# 2. 結果の表示
print("\n" + "="*30)
print(f"【明日の予測結果: {ticker}】")
print(f"上昇する確率: {prediction_prob * 100:.2f}%")

if prediction_prob > 0.5:
    print("判定: ★上昇（買いシグナル）")
else:
    print("判定: ☆下落（売りシグナル）")
print("="*30)