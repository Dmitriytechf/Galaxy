import numpy as np
import matplotlib.pyplot as plt


stars = 80000 # кол-во звезд
arms = 8 #рукав галактики
twist = 8 # насколько сильно рукава закручены

# Распределение по радиусу (плотнее к центру)
r = np.random.exponential(scale=2.0, size=stars)

# Спираль + шум
theta_arm = np.random.randint(0, arms, stars) * (2*np.pi/arms)
theta = theta_arm + twist * r + np.random.normal(0, 0.8, stars)

x = r * np.cos(theta)
y = r * np.sin(theta)

plt.figure(figsize=(8,8), facecolor='black', num='Статичная галактика')
plt.scatter(x, y, c=r, s=0.8, cmap='twilight', alpha=0.8)
plt.axis('equal'); plt.axis('off')
plt.show()
