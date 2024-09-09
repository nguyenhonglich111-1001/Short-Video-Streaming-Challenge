import matplotlib.pyplot as plt

# Data
old_qoe = 206
new_qoe_per_10m_training = [
    -1397, -1255, -1540, -1200, -1381,
    -1640, -1640, -1328, -1166, -1166,
    -1072, -1285
]

# Generating 10-minute intervals
time_intervals = [10 * i for i in range(1, len(new_qoe_per_10m_training) + 1)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_intervals, new_qoe_per_10m_training, marker='o', label="New QoE")
plt.axhline(y=old_qoe, color='r', linestyle='--', label="Old QoE")

# Adding labels and title
plt.title("Comparison of New QoE per 10m Training to Old QoE")
plt.xlabel("Time (minutes)")
plt.ylabel("QoE")
plt.xticks(time_intervals)
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
