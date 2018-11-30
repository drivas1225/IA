from random import randint

import numpy as np
from matplotlib import pyplot as plt

from hopfieldnet.net import HopfieldNetwork
from hopfieldnet.trainers import hebbian_training

# Create the training patterns
uno_pattern = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 1, 1, 1]])

dos_pattern = np.array([[0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1]])

tres_pattern = np.array([[0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0]])

cuat_pattern = np.array([[0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1 ,0, 1, 0],
                        [1, 0 ,0, 1, 0],
                        [1, 1 ,1, 1, 1],
                        [0, 0 ,0, 1, 0],
                        [0, 0 ,0, 1, 0]])

cinco_pattern = np.array([[1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0]])

seis_pattern = np.array([[0, 0, 1, 1, 1],
                        [0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0]])

siete_pattern = np.array([[1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0]])
        
ocho_pattern = np.array([[0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0]])
        
nueve_pattern = np.array([[0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0],
                        [0, 1, 1, 0, 0]])
        
cero_pattern = np.array([[0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0]])
 



uno_pattern *= 2
uno_pattern -= 1

dos_pattern *= 2
dos_pattern -= 1

tres_pattern *= 2
tres_pattern -= 1

cuat_pattern *= 2
cuat_pattern -= 1

cinco_pattern *= 2
cinco_pattern -= 1

seis_pattern *= 2
seis_pattern -= 1

siete_pattern *= 2
siete_pattern -= 1

ocho_pattern *= 2
ocho_pattern -= 1

nueve_pattern *= 2
nueve_pattern -= 1

cero_pattern *= 2
cero_pattern -= 1

input_patterns = np.array([uno_pattern.flatten(), 
                          dos_pattern.flatten(), 
                          tres_pattern.flatten(), 
                          cuat_pattern.flatten(), 
                          ])




network = HopfieldNetwork(35)

hebbian_training(network, input_patterns)


uno_test = uno_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    uno_test[p] *= -1

uno_result = network.run(uno_test)

uno_result.shape = (7, 5)
uno_test.shape = (7, 5)


dos_test = dos_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    dos_test[p] *= -1

dos_result = network.run(dos_test)

dos_result.shape = (7, 5)
dos_test.shape = (7, 5)

tres_test = tres_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    tres_test[p] *= -1

tres_result = network.run(tres_test)

tres_result.shape = (7, 5)
tres_test.shape = (7, 5)

cuat_test = cuat_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    cuat_test[p] *= -1

cuat_result = network.run(cuat_test)

cuat_result.shape = (7, 5)
cuat_test.shape = (7, 5)

cinco_test = cinco_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    cinco_test[p] *= -1

cinco_result = network.run(cinco_test)

cinco_result.shape = (7, 5)
cinco_test.shape = (7, 5)

seis_test = seis_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    seis_test[p] *= -1

seis_result = network.run(seis_test)

seis_result.shape = (7, 5)
seis_test.shape = (7, 5)

ocho_test = ocho_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    ocho_test[p] *= -1

ocho_result = network.run(ocho_test)

ocho_result.shape = (7, 5)
ocho_test.shape = (7, 5)

nueve_test = nueve_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    nueve_test[p] *= -1

nueve_result = network.run(nueve_test)

nueve_result.shape = (7, 5)
nueve_test.shape = (7, 5)

cero_test = cero_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    cero_test[p] *= -1

cero_result = network.run(cero_test)

cero_result.shape = (7, 5)
cero_test.shape = (7, 5)


# Show the results
plt.subplot(5, 2, 1)
plt.imshow(uno_test, interpolation="nearest")
plt.subplot(5, 2, 2)
plt.imshow(uno_result, interpolation="nearest")

plt.subplot(5, 2, 3)
plt.imshow(dos_test, interpolation="nearest")
plt.subplot(5, 2, 4)
plt.imshow(dos_result, interpolation="nearest")

plt.subplot(5, 2, 5)
plt.imshow(tres_test, interpolation="nearest")
plt.subplot(5, 2, 6)
plt.imshow(tres_result, interpolation="nearest")

plt.subplot(5, 2, 7)
plt.imshow(cuat_test, interpolation="nearest")
plt.subplot(5, 2, 8)
plt.imshow(cuat_result, interpolation="nearest")

plt.subplot(5, 2, 9)
plt.imshow(cero_test, interpolation="nearest")
plt.subplot(5, 2, 10)
plt.imshow(cero_result, interpolation="nearest")


plt.show()
