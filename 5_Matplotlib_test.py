import numpy as np
import matplotlib.pyplot as plt

# Standard canvas for 3 further plots
# fig, ax = plt.subplots()

# Scatter
# X = np. random.uniform(0, 1, 100)
# Y = np. random.uniform(0, 1, 100)
# ax.scatter(X, Y)

# Imshow
# Z = np. random.uniform(0, 1, (8, 8))
# ax.imshow(Z)

# Plot, line styles
# X = np. linspace(0, 10, 100)
# Y = np. sin(X)
# ax. plot(
#     X, Y,
#     color="black",
#     linestyle="--",
#     # linewidth=5,
#     # marker="o",
# )

# Plot several data
X = np.linspace(0, 10, 100)
Y1, Y2 = np.sin(X), np.cos(X)
# Two graphs on one canvas
# fig, ax = plt.subplots()
# ax.plot(X, Y1, X, Y2)

# Two rows
fig, (ax1, ax2) = plt. subplots(2, 1)
ax1.plot(X, Y1, color="C1")
ax1.set_title("Sine Wave")
ax2.plot(X, Y2, color="C0")
ax2.set_title("Cos Wave")

# Two cols
# fig, (ax1, ax2) = plt. subplots(1, 2)
# ax1.plot(Y1, X, color="C1")
# ax2.plot(Y2, X, color="C0")

plt.show()
