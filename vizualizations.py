import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/ai4i2020.csv")

# Failure distribution
sns.countplot(x="Machine failure", data=df)
plt.title("Machine Failure Distribution")
plt.savefig("failure_distribution.png")
plt.clf()

# Sensor trends before failure
df_fail = df[df["Machine failure"] == 1].head(200)
df_fail[
    ["Air temperature [K]", "Process temperature [K]"]
].plot()
plt.title("Sensor Trends Before Failure")
plt.savefig("sensor_trends.png")
plt.clf()
