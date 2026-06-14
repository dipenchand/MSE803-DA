import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("Data_set_w1A1.xlsx", sheet_name="descriptive_aggregation (1)")

# Sort by highest sales first
df = df.sort_values("sales_sum", ascending=False)

total_sales = df["sales_sum"].sum()
total_quantity = df["quantity_sum"].sum()

print("=== Summary ===")
print("Total sales:", round(total_sales, 2))
print("Total quantity:", int(total_quantity))
print("Top sales category:", df.loc[df["sales_sum"].idxmax(), "category"])

print("\nSales share by category:")
for _, row in df.iterrows():
    share = (row["sales_sum"] / total_sales) * 100
    print(f"- {row['category']}: {share:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(df["category"], df["sales_sum"], color="skyblue")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.subplot(1, 2, 2)
plt.bar(df["category"], df["quantity_sum"], color="lightgreen")
plt.title("Quantity by Category")
plt.xlabel("Category")
plt.ylabel("Quantity")

plt.tight_layout()
plt.savefig("sales_quantity_by_category.png", dpi=150)
print("\nPlot saved: sales_quantity_by_category.png")
