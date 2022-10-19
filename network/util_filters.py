import math
import cv2
import torch
import torch.nn as nn

# import trilinear
import numpy as np
import torchvision.models as models

import os
import sys
'''
output states:
    0: has rewards?
    1: stopped?
    2: num steps
    3:
'''
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3








def make_image_grid(images, per_row=8, padding=2):
  npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
  images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
  assert images.shape[0] % per_row == 0
  num_rows = images.shape[0] // per_row
  image_rows = []
  for i in range(num_rows):
    image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
  return np.vstack(image_rows)


def get_image_center(image):
  if image.shape[0] > image.shape[1]:
    start = (image.shape[0] - image.shape[1]) // 2
    image = image[start:start + image.shape[1], :]

  if image.shape[1] > image.shape[0]:
    start = (image.shape[1] - image.shape[0]) // 2
    image = image[:, start:start + image.shape[0]]
  return image


def rotate_image(image, angle):
  """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) // 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5

  # Obtain the rotated coordinates of the image corners
  rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
  ]

  # Find the size of the new image
  x_coords = [pt[0] for pt in rotated_coords]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]

  y_coords = [pt[1] for pt in rotated_coords]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]

  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)

  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))

  # We require a translation matrix to keep the image centred
  trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                         [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

  # Compute the tranform for the combined rotation and translation
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

  # Apply the transform
  result = cv2.warpAffine(
      image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

  return result







# def lrelu(x, leak=0.2, name="lrelu"):
#   with tf.variable_scope(name):
#     f1 = 0.5 * (1 + leak)
#     f2 = 0.5 * (1 - leak)
#     return f1 * x + f2 * abs(x)





def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                                  1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]


def tanh01(x):
  # return tf.tanh(x) * 0.5 + 0.5
  return torch.tanh(x) * 0.5 + 0.5



def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)




def lerp(a, b, l):
  return (1 - l) * a + l * b

# class Generator3DLUT_identity(nn.Module):
#     def __init__(self, dim=33):
#         super(Generator3DLUT_identity, self).__init__()
#         if dim == 33:
#             file = open("IdentityLUT33.txt", 'r')
#         elif dim == 64:
#             file = open("IdentityLUT64.txt", 'r')
#         lines = file.readlines()
#         buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)
#
#         for i in range(0, dim):
#             for j in range(0, dim):
#                 for k in range(0, dim):
#                     n = i * dim * dim + j * dim + k
#                     x = lines[n].split()
#                     buffer[0, i, j, k] = float(x[0])
#                     buffer[1, i, j, k] = float(x[1])
#                     buffer[2, i, j, k] = float(x[2])
#         self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
#         self.TrilinearInterpolation = TrilinearInterpolation()
#
#     def forward(self, x):
#         _, output = self.TrilinearInterpolation(self.LUT, x)
#         # self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
#         return output
#

# class Generator3DLUT_zero(nn.Module):
#     def __init__(self, dim=33):
#         super(Generator3DLUT_zero, self).__init__()
#
#         self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
#         # self.LUT = nn.Parameter(torch.tensor(self.LUT))
#         self.LUT = nn.Parameter(self.LUT.clone().detach().requires_grad_(True))
#
#
#         self.TrilinearInterpolation = TrilinearInterpolation()
#
#     def forward(self, x):
#         _, output = self.TrilinearInterpolation(self.LUT, x)
#
#         return output
#
#
# class TrilinearInterpolationFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, lut, x):
#         x = x.contiguous()
#
#         output = x.new(x.size())
#         dim = lut.size()[-1]
#         shift = dim ** 3
#         binsize = 1.000001 / (dim - 1)
#         W = x.size(2)
#         H = x.size(3)
#         batch = x.size(0)
#
#         assert 1 == trilinear.forward(lut,
#                                       x,
#                                       output,
#                                       dim,
#                                       shift,
#                                       binsize,
#                                       W,
#                                       H,
#                                       batch)
#
#         int_package = torch.IntTensor([dim, shift, W, H, batch])
#         float_package = torch.FloatTensor([binsize])
#         variables = [lut, x, int_package, float_package]
#
#         ctx.save_for_backward(*variables)
#
#         return lut, output
#
#     @staticmethod
#     def backward(ctx, lut_grad, x_grad):
#         lut, x, int_package, float_package = ctx.saved_variables
#         dim, shift, W, H, batch = int_package
#         dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
#         binsize = float(float_package[0])
#
#         assert 1 == trilinear.backward(x,
#                                        x_grad,
#                                        lut_grad,
#                                        dim,
#                                        shift,
#                                        binsize,
#                                        W,
#                                        H,
#                                        batch)
#         return lut_grad, x_grad
#
#
# class TrilinearInterpolation(torch.nn.Module):
#     def __init__(self):
#         super(TrilinearInterpolation, self).__init__()
#
#     def forward(self, lut, x):
#         return TrilinearInterpolationFunction.apply(lut, x)
#
#
# class TV_3D(nn.Module):
#     def __init__(self, dim=33):
#         super(TV_3D, self).__init__()
#
#         self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
#         self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
#         self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
#         self.weight_g[:, :, (0, dim - 2), :] *= 2.0
#         self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
#         self.weight_b[:, (0, dim - 2), :, :] *= 2.0
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, LUT):
#         dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
#         dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
#         dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
#         tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
#             torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))
#
#         mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))
#
#         return tv, mn
#
#
