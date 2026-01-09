from matplotlib import pyplot as plt

multiplier = 50

plt.scatter(x=[0,1], y=[0,1], c=[0,100/multiplier], cmap="jet")
plt.colorbar(label="Chamfer distance (cm)", orientation="vertical")
plt.show()

plt.scatter(x=[0,1], y=[0,1], c=[0,100/multiplier], cmap="jet")
plt.colorbar(label="Chamfer distance (cm)", orientation="horizontal")
plt.show()
