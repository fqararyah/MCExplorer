import random
import time

a = 0
for i in range(1000):
    a += random.randint(0, 10)
    time.sleep(10)

print(a)