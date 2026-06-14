import numpy as np
from scipy.signal import convolve2d, correlate2d

A = np.array([[1, 2, 3],
              [5, 6, 7],
              [10, 0, 11]])

B = np.array([[5, 3],
              [9, 1]])

print("Input matrix A:")
print(A)
print("\nKernel B:")
print(B)

# True 2D convolution (kernel is flipped 180° before sliding)
conv_result = convolve2d(A, B, mode='valid')
print("\n--- True Convolution (convolve2d, mode='valid') ---")
print("Flipped kernel (B rotated 180°):")
print(np.flip(B))
print("Result:")
print(conv_result)

# Cross-correlation (no kernel flip — used in most ML frameworks)
xcorr_result = correlate2d(A, B, mode='valid')
print("\n--- Cross-Correlation (correlate2d, mode='valid') ---")
print("Result:")
print(xcorr_result)