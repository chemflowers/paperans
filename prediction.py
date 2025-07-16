import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("cvpr_titles.csv")

year_counts = df['year'].value_counts().sort_index()
year_counts.index = year_counts.index.astype(int)

model = ARIMA(year_counts.values, order=(1, 1, 1))  # 你可以调试 order=(p,d,q)
model_fit = model.fit()

forecast = model_fit.forecast(steps=1)
predicted_2025 = int(forecast[0])

last_year = year_counts.index.max()
next_year = last_year + 1

print(f"Predicted paper count for {next_year}: {predicted_2025}")

plt.figure(figsize=(10, 6))
plt.plot(year_counts.index, year_counts.values, marker='o', color='teal', label='Actual')

plt.scatter([next_year], [predicted_2025], color='red', marker='o', s=150, label='Prediction (2025)')
plt.text(next_year, predicted_2025 + 10, f"{predicted_2025}", ha='center', fontsize=12, color='red')

plt.title("CVPR Paper Count Trend (2020-2025)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.xticks(list(year_counts.index) + [next_year])
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("cvpr_forecast_2025.png", dpi=300)

plt.show()

