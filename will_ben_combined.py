import numpy as np
import pywt
import math
import matplotlib.pyplot as plt

def bensEntropy(image):
    np.set_printoptions(threshold=np.nan)
    entropyArray = np.zeros((len(image) - 1, len(image) - 1))
    
    for i in range(len(image) - 1):
        for j in range(len(image) - 1):
            subArray = np.array([[image[i][j], image[i][j + 1]],[image[i + 1][j], image[i + 1][j + 1]]])
            entropyArray[i][j] = getEntropy(subArray)
    return entropyArray

def getEntropy(m):
    num, freq = np.unique(m, return_counts=True)
    freq = freq / freq.sum()
    entropy = 0
    for f in freq:
        f1 = f
        entropy -= (f1*math.log(f1, 2))
    return entropy

# turn the text file into a numpy matrix
image = np.loadtxt('AFMimage1.txt')

# multilevel wavelet decomposition
coeffs = pywt.wavedec2(image, 'haar')
levels = len(coeffs)-1

# run the transformed image through ben's entropy thing, then plot the entropies
horizontal = []
vertical = []
diagonal = []
xAxis = []

for index in range(1, levels + 1): 
    cH, cV, cD = coeffs[index]
    cH = bensEntropy(cH)
    cV = bensEntropy(cV)
    cD = bensEntropy(cD)
    cHentropy = getEntropy(cH)
    cVentropy = getEntropy(cV)
    cDentropy = getEntropy(cD)
    horizontal.append(cHentropy)
    vertical.append(cVentropy)
    diagonal.append(cDentropy)
    xAxis.append(levels + 1 - index)

# plot the different entropies
plt.figure(1)
plt.subplot(511)
plt.plot(xAxis, horizontal, 'r')
plt.ylabel('Horizontal Entropy')
plt.subplot(513)
plt.plot(xAxis, vertical, 'b')
plt.ylabel('Vertical Entropy')
plt.subplot(515)
plt.plot(xAxis, diagonal, 'g')
plt.ylabel('Diagonal Entropy')
plt.xlabel('Level')
plt.show()
