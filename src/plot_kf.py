import numpy as np
import matplotlib.pyplot as plt

plt.ion()
data = (center_red, center_green, center_blue)
# colors = ("red", "green", "blue")
colors = ('r--', 'g-', 'bs')
groups = ("Prediction", "Detection", "Update")

# Create plot
fig = plt.figure()
ax = fig.subplots(1, 1)
for cent, color, group in zip(data, colors, groups):
    x_only, y_only = zip(*cent)
    ax.plot(x_only, y_only, color,  alpha=0.8, label=group)

plt.title('Kalman Kilter')
plt.legend(loc=2)
plt.savefig("kf2.png")
plt.show()
