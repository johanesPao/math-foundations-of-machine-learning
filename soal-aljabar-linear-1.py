# Jill mendesain solar panel sebagai hobby
# Pada 1 April, Mark I milik Jill mulai menghasilkan energi sebanyak 1kJ/hari
# Pada 1 May, Mark II milik Jill mulai menghasilkan energi sebesar 4 kJ/hari

# 1. Pada hari keberapa Mark II milik Jill menghasilkan total energi sebanyak Mark I?
# 2. Berapa banyak total energi yang dihasilkan pada hari tersebut?
# 3. Bagaimana solusi di pertanyaan pertama dan kedua jika Mark II didesain untuk menghasilkan energi sebanyak 1kJ/hari?

# total energi dimana Mark II menghasilkan energi sama dengan Mark II adalah:
# e = 1 * h (Persamaan I untuk Mark I)
# e = 4(h - 30) (Persamaan II untuk Mark II)

# e = e
# 1 * h = 4h - 120
# h = 4h - 120
# -3h = -120
# h = 40

# e = 1 * 40
# e = 40

# Jawaban:
# 1. Pada hari ke-40 Mark II dan Mark I menghasilkan jumlah energi yang sama
# 2. Total energi yang dihasilkan oleh keduanya pada hari tersebut adalah 80kJ (40kJ + 40kJ)
# 3. Tidak ada solusi karena garis persamaan tidak akan pernah bersinggungan

import numpy as np
import matplotlib.pyplot as plt

h = np.linspace(0, 50, 1000)
e_mk1 = 1 * h
e_mk2 = 4 * (h - 30)

fig, ax = plt.subplots()
plt.title('Jumlah Hari dan Energi Mark I - Mark II')
plt.xlabel('Waktu (dalam hari)')
plt.ylabel('Energi (dalam kJ)')
ax.set_xlim([0, 50])
ax.set_ylim([0, 50])
ax.plot(h, e_mk1, c='red')
ax.plot(h, e_mk2, c='green')
plt.axvline(x=40, color='purple', linestyle='--')
_ = plt.axhline(y=40, color='purple', linestyle='--')
plt.show()