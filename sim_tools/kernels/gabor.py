from ..common import *


def gabor_cos(x, y, f, a):
  return np.cos(2.*np.pi*f*(x*np.cos(a) + y*np.sin(a)))


def gabor_sin(x, y, f, a):
  return np.sin(2.*np.pi*f*(x*np.cos(a) + y*np.sin(a)))


def gabor_exp(x, y, s):
  return np.exp(-(x**2 + y**2)/(2*s**2))


def gabor_xy(kernel_width):
  half_width = kernel_width//2
  xmax = half_width; ymax = half_width
  x, y = np.meshgrid(np.arange(-xmax, xmax + 1),
                     np.arange(-ymax, ymax + 1))
  y = -y
  return x, y


def gabor(kernel_width, angle, sigma, freq):
  x, y = gabor_xy(kernel_width)
  gc = gabor_cos(x, y, freq, angle)
  gs = gabor_sin(x, y, freq, angle)
  ge = gabor_exp(x, y, sigma)
  gbr = gc*gs*ge
  gbr = sum2zero(gbr)
  gbr,w = conv2one(gbr)

  return gbr


def multi_gabor(kernel_width, angles, sigma, freq):
  tmp_k = [gabor(kernel_width, a, sigma, freq)  for a in angles]
  
  similar = []
  for i in range(len(angles)):
    for j in range(i+1, len(angles)):
      error = np.mean((tmp_k[i] - tmp_k[j])**2)
      if error <= 0.01:
        similar.append(j)
  
  kernels = [tmp_k[i] for i in range(len(angles)) if i not in similar]

  return kernels
  
