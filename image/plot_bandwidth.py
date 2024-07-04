import matplotlib.pyplot as plt

# Read the data from the file
file_path = "./0"
data_from_file = []

with open(file_path, "r") as file:
    for line in file:
        time, bandwidth = map(float, line.split())
        data_from_file.append((time, bandwidth))

# Separate the data into x and y components
x_from_file, y_from_file = zip(*data_from_file[:50])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_from_file, y_from_file, marker='o', linestyle='-', color='b')
plt.title('Bandwidth Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Bandwidth (Mbps)')
plt.grid(True)
plt.show()
