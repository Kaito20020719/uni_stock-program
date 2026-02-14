import yfinance as yf
import matplotlib.pyplot as plt

ticker = "9983.T"
data = yf.download(ticker , period = "1d",interval="1m")

fig,(ax1,ax2) = plt.subplots(2,1,figsize = (10,8),sharex = True)

ax1.plot(data.index,data["Close"],label = "Stock Price",color = "blue")
ax1.set_title(f"{ticker} Stock Price and Volume")
ax1.set_ylabel("Price(JPY)")
ax1.grid(True)


ax2.bar(data.index,data["Volume"].values.flatten(),label = "Volume",color = "gray",width = 0.0005)
ax2.grid(True)

plt.tight_layout()
plt.show()

