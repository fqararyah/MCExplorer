import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

fig = plt.figure(layout='constrained')
sf1, sf2 = fig.subfigures(1, 2, wspace=0.07)

ax1a, ax1b = sf1.subplots(2, 1)
ax1a.scatter(0, 0)
ax1b.scatter(0, 0)

ax2 = sf2.subplots()
ax2.scatter(0, 0)

conn = ConnectionPatch(
    xyA=(0, 0), coordsA='data', axesA=ax1a,
    xyB=(0, 0), coordsB='data', axesB=ax2,
    color='red',
)
ax2.add_artist(conn)
conn.set_in_layout(False) # remove from layout calculations

plt.show()