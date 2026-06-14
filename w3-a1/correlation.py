from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("age_networth.xlsx")
x = df.iloc[:, 0]
y = df.iloc[:, 1]

r, _ = stats.pearsonr(x, y)
print(f"Pearson correlation coefficient: {r:.4f}")

slope, intercept, *_ = stats.linregress(x, y)

plt.scatter(x, y, label="Data")
plt.plot(x, slope * x + intercept, color="red", label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title(f"Correlation: r = {r:.4f}")
plt.legend()
plt.show()
