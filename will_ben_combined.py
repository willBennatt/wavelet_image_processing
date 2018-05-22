import numpy as np
import pywt
import math

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

cH7, cV7, cD7 = coeffs[3]
cH3, cV3, cD3 = coeffs[7]

# do whatever with the coefficients
print('entropy before: ' + str(getEntropy(cH3)))
entropyArray = bensEntropy(cH3)
print('entropy after: ' + str(getEntropy(entropyArray)))
