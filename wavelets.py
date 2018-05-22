import numpy as np
import pywt
from PIL import Image
from sklearn import preprocessing as skp
import math
import matplotlib.pyplot as plt

# turn the text file into a numpy matrix
image = np.loadtxt('AFMimage1.txt')
# scaler to use for all images
scaler = skp.MinMaxScaler(feature_range=(0, 250))

# method to display a matrix as an image
def displayMatrix(m):
    im = Image.fromarray(m)
    im.show()

# method to scale a matrix so the format will be better for displaying
def scaleMatrix(m):
    scaler1 = scaler.fit(m)
    m_scaled = scaler1.transform(m)
    return m_scaled

# This method actually works and returns the entropy for a matrix
# Works for 1d or 2d matrices
def getEntropy(m):
    num, freq = np.unique(m, return_counts=True)
    freq = freq / freq.sum()
    entropy = 0
    sum = 0
    for f in freq:
        f1 = f
        sum += f1
        entropy -= (f1*math.log(f1, 2))
    #print(sum)
    return entropy
               

# display unscaled image
#displayMatrix(image)

# display scaled image
image_scaled = scaleMatrix(image)
displayMatrix(image_scaled)

# now do a db1 wavelet transform on the image
#(cA, cB) = pywt.dwt(image, 'db1')

# 2d array transform
coeffs = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs

# Testing my Shannon Entropy functions
#x = np.array([1, 2, 3, 4])
#print(getEntropy3(x))

# scale and display each directional transform in coeffs, 
# then print the entropy of each array 
'''
for array in cA, cH, cV, cD:
    array_scaled = scaleMatrix(array)
    #displayMatrix(array_scaled)
    print('entropy: ' + str(getEntropy(array)))
    '''

# multilevel wavelet decomposition
coeffs2 = pywt.wavedec2(image, 'haar')
# Levels
levels = len(coeffs2)-1
print('Total Levels: ' + str(levels))

# Analyze the overall approximation
cA = coeffs2[0]
print()
print('Overall approximation: ' + str(cA))
print('Entropy of cA: ' + str(getEntropy(cA)))
print()

horizontal = []
vertical = []
diagonal = []
xAxis = []

# Analyze the entropy of each level
for index in range(1, levels + 1): 
    cH, cV, cD = coeffs2[index]
    cHentropy = getEntropy(cH)
    cVentropy = getEntropy(cV)
    cDentropy = getEntropy(cD)
    print('Level: ' + str(levels + 1 - index))
    print('Entropy of cH: ' + str(cHentropy))
    print('Entropy of cV: ' + str(cVentropy))
    print('Entropy of cD: ' + str(cDentropy))
    print()
    horizontal.append(cHentropy)
    vertical.append(cVentropy)
    diagonal.append(cDentropy)
    xAxis.append(levels + 1 - index)

# displaying coarse vs fine detail images

cH9, cV9, cD9 = coeffs[1]
ch9_scaled = scaleMatrix(cH9)
displayMatrix(ch9_scaled)

cH1, cV1, cD1 = coeffs2[levels]
cH1_scaled = scaleMatrix(cH1)
displayMatrix(cH1_scaled)


# plot the different entropies
plt.plot(xAxis, horizontal, 'r')
plt.plot(xAxis, vertical, 'b')
plt.plot(xAxis, diagonal, 'g')
plt.ylabel('Entropy')
plt.xlabel('Level')
plt.show()
