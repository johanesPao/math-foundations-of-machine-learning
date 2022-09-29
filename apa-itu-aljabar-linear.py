import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 40, 1000)  # start, finish, n points

d_pb = 2.5 * t

d_p = 3 * (t - 5)

fig, ax = plt.subplots()
plt.title("Perampok Bank Tertangkap")
plt.xlabel("waktu (dalam menit)")
plt.ylabel("jarak (dalam km)")
ax.set_xlim([0, 40])
ax.set_ylim([0, 100])
ax.plot(t, d_pb, c="green")
ax.plot(t, d_p, c="brown")
plt.axvline(x=30, color="purple", linestyle="--")
_ = plt.axhline(y=75, color="purple", linestyle="--")

plt.show()
