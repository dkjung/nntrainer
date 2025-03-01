// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_neon.cpp
 * @date   4 Aug 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Source for blas neon implementation
 *
 */

#include <blas_neon.h>
#include <blas_neon_setting.h>
#include <hgemm.h>
#include <memory>
#include <nntrainer_error.h>

namespace nntrainer::neon {

void sgemv(const float *A, const float *X, float *Y, uint32_t rows,
           uint32_t cols, float alpha, float beta) {
  const float *__restrict x;

  for (unsigned int i = 0; i < rows; ++i) {
    Y[i] = Y[i] * beta;
  }

  float32x4_t v_alpha = vmovq_n_f32(alpha);

  if (cols % 16 == 0) {
    for (unsigned i = 0; i < cols; i += 16) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);
      float32x4_t x8_11 = vld1q_f32(&X[i + 8]);
      float32x4_t x12_15 = vld1q_f32(&X[i + 12]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
        x8_11 = vmulq_f32(x8_11, v_alpha);
        x12_15 = vmulq_f32(x12_15, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);
        y0 = vmlaq_f32(y0, wvec8_11, x8_11);
        y0 = vmlaq_f32(y0, wvec12_15, x12_15);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }

  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  } else if (cols % 4 == 0) {
    for (unsigned i = 0; i < cols; i += 4) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  }
}

void sgemv_transpose(const float *A, const float *X, float *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta) {
  const float *__restrict x;

  const float32x4_t v_beta = vdupq_n_f32(beta);
  const float32x4_t v_alpha = vdupq_n_f32(alpha);

  if (cols % 16 == 0) {
    unsigned int n = cols / 16;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);

    NNTR_THROW_IF(initialized == nullptr, std::invalid_argument)
      << "Error: Memory allocation failed.";

    unsigned int step;
    for (unsigned int i = 0; i < cols / 16; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 16) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);
        float32x4_t y8_11 = vld1q_f32(&y[8]);
        float32x4_t y12_15 = vld1q_f32(&y[12]);
        step = j / 16;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          y8_11 = vmulq_f32(y8_11, v_beta);
          y12_15 = vmulq_f32(y12_15, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        y8_11 = vmlaq_f32(y8_11, wvec8_11, x);
        y12_15 = vmlaq_f32(y12_15, wvec12_15, x);

        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
        vst1q_f32(&y[8], y8_11);
        vst1q_f32(&y[12], y12_15);
      }
    }
    free(initialized);
    return;
  } else if (cols % 8 == 0) {
    unsigned int n = cols / 8;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);

    NNTR_THROW_IF(initialized == nullptr, std::invalid_argument)
      << "Error: Memory allocation failed.";

    unsigned int step;
    for (unsigned int i = 0; i < cols / 8; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 8) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);

        step = j / 8;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
      }
    }
    free(initialized);
    return;
  } else if (cols % 4 == 0) {
    unsigned int n = cols / 4;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);

    NNTR_THROW_IF(initialized == nullptr, std::invalid_argument)
      << "Error: Memory allocation failed.";

    unsigned int step;
    for (unsigned int i = 0; i < cols / 4; ++i) {
      initialized[i] = false;
    }
    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 4) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        step = j / 4;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        vst1q_f32(&y[0], y0_3);
      }
    }
    free(initialized);
  }

  return;
}

void copy_int4_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {

  unsigned int idx = 0;

  // keep in mind that : len(X) = N, and len(Y) = 2*N

  // processing batch of 16
  float32x4_t y0, y1, y2, y3;
  float32x4_t y4, y5, y6, y7;

  uint8_t low0, low1, high0, high1;

  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);

    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);
    unsigned int i = 0;
    for (; i < 8; ++i) {
      low0 = low[i] >> 4;
      low1 = low[i] & 0x0f;

      high0 = high[i] >> 4;
      high1 = high[i] & 0x0f;

      // 0 ~ 8
      if (i < 2) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else if (i < 4) {
        y1[2 * (i - 2)] = low0;
        y1[2 * (i - 2) + 1] = low1;
      } else if (i < 6) {
        y2[2 * (i - 4)] = low0;
        y2[2 * (i - 4) + 1] = low1;
      } else {
        y3[2 * (i - 6)] = low0;
        y3[2 * (i - 6) + 1] = low1;
      }

      // 8 ~ 16
      if (i < 2) {
        y4[2 * i] = high0;
        y4[2 * i + 1] = high1;
      } else if (i < 4) {
        y5[2 * (i - 2)] = high0;
        y5[2 * (i - 2) + 1] = high1;
      } else if (i < 6) {
        y6[2 * (i - 4)] = high0;
        y6[2 * (i - 4) + 1] = high1;
      } else {
        y7[2 * (i - 6)] = high0;
        y7[2 * (i - 6) + 1] = high1;
      }
    }
    vst1q_f32(&Y[2 * idx], y0);
    vst1q_f32(&Y[2 * idx + 4], y1);
    vst1q_f32(&Y[2 * idx + 8], y2);
    vst1q_f32(&Y[2 * idx + 12], y3);
    vst1q_f32(&Y[2 * idx + 16], y4);
    vst1q_f32(&Y[2 * idx + 20], y5);
    vst1q_f32(&Y[2 * idx + 24], y6);
    vst1q_f32(&Y[2 * idx + 28], y7);
  }

  // processing remaining batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    unsigned int i = 0;
    for (; i < 8; ++i) {
      low0 = batch[i] >> 4;
      low1 = batch[i] & 0x0f;

      if (i < 2) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else if (i < 4) {
        y1[2 * (i - 2)] = low0;
        y1[2 * (i - 2) + 1] = low1;
      } else if (i < 6) {
        y2[2 * (i - 4)] = low0;
        y2[2 * (i - 4) + 1] = low1;
      } else {
        y3[2 * (i - 6)] = low0;
        y3[2 * (i - 6) + 1] = low1;
      }
    }

    vst1q_f32(&Y[2 * idx], y0);
    vst1q_f32(&Y[2 * idx + 4], y1);
    vst1q_f32(&Y[2 * idx + 8], y2);
    vst1q_f32(&Y[2 * idx + 12], y3);
  }

  // pocessing remaining values
  for (; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void copy_int8_or_int4(const unsigned int N, const uint8_t *X, uint8_t *Y) {
  ///@note int8 Tensor and int4 Tensor share the same memory offset
  unsigned int idx = 0;
  for (; N - idx >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    vst1q_u8(&Y[idx], batch);
  }
  for (; N - idx >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);
    vst1_u8(&Y[idx], batch);
  }
  for (; N - idx >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_int8(const unsigned int N, const int8_t *X, int8_t *Y) {
  ///@note int8 Tensor and int4 Tensor share the same memory offset
  unsigned int idx = 0;
  for (; N - idx >= 16; idx += 16) {
    int8x16_t batch = vld1q_s8(&X[idx]);
    vst1q_s8(&Y[idx], batch);
  }
  for (; N - idx >= 8; idx += 8) {
    int8x8_t batch = vld1_s8(&X[idx]);
    vst1_s8(&Y[idx], batch);
  }
  for (; N - idx >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void sine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    if (alpha != 1.0)
      x0_3 = vmulq_n_f32(x0_3, alpha);
    float32x4_t sinx0_3 = sin_ps(x0_3);
    vst1q_f32(&Y[i], sinx0_3);
  }
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]);
    ++i;
  }
}

void cosine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    if (alpha != 1.0)
      x0_3 = vmulq_n_f32(x0_3, alpha);
    float32x4_t cosx0_3 = cos_ps(x0_3);
    vst1q_f32(&Y[i], cosx0_3);
  }
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]);
    ++i;
  }
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_7 = vld1q_f32(&X[i]);
    float32x4_t x0_7_sqrt = vsqrtq_f32(x0_7);
    float32x4_t ones = vmovq_n_f32(1);
    float32x4_t x0_7_sqrt_div = vdivq_f32(ones, x0_7_sqrt);
    vst1q_f32(&X[i], x0_7_sqrt_div);
  }
  while (i < N) {
    X[i] = (1 / std::sqrt(static_cast<float>(X[i])));
    ++i;
  }
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (alpha != 1.f) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vmulq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = alpha * X[i] * Y[i] + beta * Z[i];
    else
      Z[i] = alpha * X[i] * Y[i];
    ++i;
  }
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (alpha != 1.f) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vaddq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] + alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] + alpha * Y[i];
    ++i;
  }
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (alpha != 1.f) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vsubq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] - alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] - alpha * Y[i];
    ++i;
  }
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (alpha != 1.f) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vdivq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] / (alpha * Y[i]) + beta * Z[i];
    else
      Z[i] = X[i] / (alpha * Y[i]);
    ++i;
  }
}

bool isValid(const size_t N, const float *X) {
  size_t i = 0;
  float inf_s = std::numeric_limits<float>::infinity();
  float32x4_t inf = vdupq_n_f32(inf_s);
  uint16x8_t zero = vdupq_n_f32(0);

  for (; N - i >= 4; i += 4) {
    float32x4_t vec = vld1q_f32(&X[i]);
    uint32x4_t vcmp = vceqq_f32(vec, vec);

    vcmp = vceqq_f32(vcmp, zero);

    if (vaddvq_u32(vcmp))
      return false;

    vcmp = vceqq_f32(vec, inf);

    if (vaddvq_u16(vcmp))
      return false;
  }

  while (i < N) {
    if (X[i] != X[i] || X[i] == std::numeric_limits<float>::infinity())
      return false;
    ++i;
  }

  return true;
}

static inline void __scopy_kernel(const float *X, float *Y) {
  vst1q_f32(Y, vld1q_f32(X));
}

void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY) {
  unsigned int N4 = (N >> 2) << 2;
  for (unsigned int i = 0; i < N4; i += 4) {
#ifdef __aarch64__
    __asm__ __volatile__("ld1 {v0.4s}, [%1]\n\t"
                         "st1 {v0.4s}, [%0]\n\t"
                         :
                         : "r"(&Y[i]), "r"(&X[i])
                         : "v0", "memory");
#else
    __scopy_kernel(X + i, Y + i);
#endif
  }
  for (unsigned int i = N4; i < N; ++i) {
    Y[i] = X[i];
  }
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  unsigned int i = 0;
  unsigned int N8 = (N >> 3) << 3;
  unsigned int N16 = (N >> 4) << 4;
  for (; i < N16; i += 16) {
    int16x8x2_t vec = vld1q_s16_x2(X);
    vst1q_s16_x2(Y, vec);
    X += 16;
    Y += 16;
  }
  for (; i < N8; i += 8) {
    int16x8_t vec = vld1q_s16(X);
    vst1q_s16(Y, vec);
    X += 8;
    Y += 8;
  }
  for (; i < N; ++i) {
    *Y = *X;
    X++;
    Y++;
  }
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  unsigned int i = 0;
  unsigned int N8 = (N >> 3) << 3;
  unsigned int N16 = (N >> 4) << 4;
  for (; i < N16; i += 16) {
    uint16x8x2_t vec = vld1q_u16_x2(X);
    vst1q_u16_x2(Y, vec);
    X += 16;
    Y += 16;
  }
  for (; i < N8; i += 8) {
    uint16x8_t vec = vld1q_u16(X);
    vst1q_u16(Y, vec);
    X += 8;
    Y += 8;
  }
  for (; i < N; ++i) {
    *Y = *X;
    X++;
    Y++;
  }
}

void copy_u8_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    // convert to u16
    uint16x8_t batch_low_u16 = vmovl_u8(low);
    uint16x8_t batch_high_u16 = vmovl_u8(high);

    // convert to u32
    uint32x4_t batch_low_u32_low = vmovl_u16(vget_low_u16(batch_low_u16));
    uint32x4_t batch_low_u32_high = vmovl_u16(vget_high_u16(batch_low_u16));
    uint32x4_t batch_high_u32_low = vmovl_u16(vget_low_u16(batch_high_u16));
    uint32x4_t batch_high_u32_high = vmovl_u16(vget_high_u16(batch_high_u16));

    // todo : experiment with vcvt_f32_u32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f32(&Y[idx], vcvtq_f32_u32(batch_low_u32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_u32(batch_low_u32_high));
    vst1q_f32(&Y[idx + 8], vcvtq_f32_u32(batch_high_u32_low));
    vst1q_f32(&Y[idx + 12], vcvtq_f32_u32(batch_high_u32_high));
  }
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    // convert to u16
    uint16x8_t batch_u16 = vmovl_u8(batch);

    // convert to u32
    uint32x4_t batch_u32_low = vmovl_u16(vget_low_u16(batch_u16));
    uint32x4_t batch_u32_high = vmovl_u16(vget_high_u16(batch_u16));

    vst1q_f32(&Y[idx], vcvtq_f32_u32(batch_u32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_u32(batch_u32_high));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_s8_to_fp32(const unsigned int N, const int8_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    int8x16_t batch = vld1q_s8(&X[idx]);
    int8x8_t low = vget_low_s8(batch);
    int8x8_t high = vget_high_s8(batch);

    // convert to s16
    int16x8_t batch_low_s16 = vmovl_s8(low);
    int16x8_t batch_high_s16 = vmovl_s8(high);

    // convert to s32
    int32x4_t batch_low_s32_low = vmovl_s16(vget_low_s16(batch_low_s16));
    int32x4_t batch_low_s32_high = vmovl_s16(vget_high_s16(batch_low_s16));
    int32x4_t batch_high_s32_low = vmovl_s16(vget_low_s16(batch_high_s16));
    int32x4_t batch_high_s32_high = vmovl_s16(vget_high_s16(batch_high_s16));

    // todo : experiment with vcvt_f32_s32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f32(&Y[idx], vcvtq_f32_s32(batch_low_s32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_s32(batch_low_s32_high));
    vst1q_f32(&Y[idx + 8], vcvtq_f32_s32(batch_high_s32_low));
    vst1q_f32(&Y[idx + 12], vcvtq_f32_s32(batch_high_s32_high));
  }
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_s8(&X[idx]);

    // convert to s16
    int16x8_t batch_s16 = vmovl_s8(batch);

    // convert to s32
    int32x4_t batch_s32_low = vmovl_s16(vget_low_s16(batch_s16));
    int32x4_t batch_s32_high = vmovl_s16(vget_high_s16(batch_s16));

    vst1q_f32(&Y[idx], vcvtq_f32_s32(batch_s32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_s32(batch_s32_high));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 8; ++idx) {
    vst1q_s16(&Y[idx],
              vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vld1q_f32(&X[idx]))),
                           vqmovn_s32(vcvtq_s32_f32(vld1q_f32(&X[idx + 4])))));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_int8_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    // convert to u16
    uint16x8_t batch_low_u16 = vmovl_u8(low);
    uint16x8_t batch_high_u16 = vmovl_u8(high);

    // convert to u32
    uint32x4_t batch_low_u32_low = vmovl_u16(vget_low_u16(batch_low_u16));
    uint32x4_t batch_low_u32_high = vmovl_u16(vget_high_u16(batch_low_u16));
    uint32x4_t batch_high_u32_low = vmovl_u16(vget_low_u16(batch_high_u16));
    uint32x4_t batch_high_u32_high = vmovl_u16(vget_high_u16(batch_high_u16));

    // todo : experiment with vcvt_f32_u32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f32(&Y[idx], vcvtq_f32_u32(batch_low_u32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_u32(batch_low_u32_high));
    vst1q_f32(&Y[idx + 8], vcvtq_f32_u32(batch_high_u32_low));
    vst1q_f32(&Y[idx + 12], vcvtq_f32_u32(batch_high_u32_high));
  }
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    // convert to u16
    uint16x8_t batch_u16 = vmovl_u8(batch);

    // convert to u32
    uint32x4_t batch_u32_low = vmovl_u16(vget_low_u16(batch_u16));
    uint32x4_t batch_u32_high = vmovl_u16(vget_high_u16(batch_u16));

    vst1q_f32(&Y[idx], vcvtq_f32_u32(batch_u32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_u32(batch_u32_high));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_int8_to_fp32(const unsigned int N, const int8_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    int8x16_t batch = vld1q_s8(&X[idx]);
    int8x8_t low = vget_low_s8(batch);
    int8x8_t high = vget_high_s8(batch);

    // convert to s16
    int16x8_t batch_low_s16 = vmovl_s8(low);
    int16x8_t batch_high_s16 = vmovl_s8(high);

    // convert to s32
    int32x4_t batch_low_s32_low = vmovl_s16(vget_low_s16(batch_low_s16));
    int32x4_t batch_low_s32_high = vmovl_s16(vget_high_s16(batch_low_s16));
    int32x4_t batch_high_s32_low = vmovl_s16(vget_low_s16(batch_high_s16));
    int32x4_t batch_high_s32_high = vmovl_s16(vget_high_s16(batch_high_s16));

    // todo : experiment with vcvt_f32_s32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f32(&Y[idx], vcvtq_f32_s32(batch_low_s32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_s32(batch_low_s32_high));
    vst1q_f32(&Y[idx + 8], vcvtq_f32_s32(batch_high_s32_low));
    vst1q_f32(&Y[idx + 12], vcvtq_f32_s32(batch_high_s32_high));
  }
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_s8(&X[idx]);

    // convert to s16
    int16x8_t batch_s16 = vmovl_s8(batch);

    // convert to s32
    int32x4_t batch_s32_low = vmovl_s16(vget_low_s16(batch_s16));
    int32x4_t batch_s32_high = vmovl_s16(vget_high_s16(batch_s16));

    vst1q_f32(&Y[idx], vcvtq_f32_s32(batch_s32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_s32(batch_s32_high));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 8; idx += 8) {
    int16x8_t batch = vld1q_s16(&X[idx]);
    int16x4_t low = vget_low_s16(batch);
    int16x4_t high = vget_high_s16(batch);

    // widen to s32
    int32x4_t low_s32 = vmovl_s16(low);
    int32x4_t high_s32 = vmovl_s16(high);

    // convert to f32
    float32x4_t low_f32 = vcvtq_f32_s32(low_s32);
    float32x4_t high_f32 = vcvtq_f32_s32(high_s32);

    vst1q_f32(&Y[idx], low_f32);
    vst1q_f32(&Y[idx + 4], high_f32);
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

#ifdef ENABLE_FP16

void hgemv(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t M, uint32_t N,
           float alpha, float beta) {
  const unsigned int batch = 0;
  const __fp16 *__restrict x;
  float *Y32 = new float[M];

  unsigned int idx = 0;

  for (; M - idx >= 8; idx += 8) {
    float32x4_t y0_3 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));
    y0_3 = vmulq_n_f32(y0_3, beta);
    y4_7 = vmulq_n_f32(y4_7, beta);

    vst1q_f32(&Y32[idx], y0_3);
    vst1q_f32(&Y32[idx + 4], y4_7);
  }
  for (; M - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_n_f32(y0_3_32, beta);

    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; idx < M; ++idx) {
    Y32[idx] = beta * Y[idx];
  }

  idx = 0;
  if (N / 120 >= batch) {
    for (; N - idx >= 120; idx += 120) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      float16x8_t x32_39 = vld1q_f16(&X[idx + 32]);
      float16x8_t x40_47 = vld1q_f16(&X[idx + 40]);
      float16x8_t x48_55 = vld1q_f16(&X[idx + 48]);
      float16x8_t x56_63 = vld1q_f16(&X[idx + 56]);

      float16x8_t x64_71 = vld1q_f16(&X[idx + 64]);
      float16x8_t x72_79 = vld1q_f16(&X[idx + 72]);
      float16x8_t x80_87 = vld1q_f16(&X[idx + 80]);

      float16x8_t x88_95 = vld1q_f16(&X[idx + 88]);
      float16x8_t x96_103 = vld1q_f16(&X[idx + 96]);
      float16x8_t x104_111 = vld1q_f16(&X[idx + 104]);
      float16x8_t x112_120 = vld1q_f16(&X[idx + 112]);

      if (alpha != 1.0) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
        x32_39 = vmulq_n_f16(x32_39, alpha);
        x40_47 = vmulq_n_f16(x40_47, alpha);
        x48_55 = vmulq_n_f16(x48_55, alpha);
        x56_63 = vmulq_n_f16(x56_63, alpha);

        x64_71 = vmulq_n_f16(x64_71, alpha);
        x72_79 = vmulq_n_f16(x72_79, alpha);
        x80_87 = vmulq_n_f16(x80_87, alpha);
        x88_95 = vmulq_n_f16(x88_95, alpha);
        x96_103 = vmulq_n_f16(x96_103, alpha);
        x104_111 = vmulq_n_f16(x104_111, alpha);
        x112_120 = vmulq_n_f16(x112_120, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        y = vfmaq_f16(y, vld1q_f16(&w[32]), x32_39);
        y = vfmaq_f16(y, vld1q_f16(&w[40]), x40_47);
        y = vfmaq_f16(y, vld1q_f16(&w[48]), x48_55);
        y = vfmaq_f16(y, vld1q_f16(&w[56]), x56_63);

        y = vfmaq_f16(y, vld1q_f16(&w[64]), x64_71);
        y = vfmaq_f16(y, vld1q_f16(&w[72]), x72_79);
        y = vfmaq_f16(y, vld1q_f16(&w[80]), x80_87);

        y = vfmaq_f16(y, vld1q_f16(&w[88]), x88_95);
        y = vfmaq_f16(y, vld1q_f16(&w[96]), x96_103);
        y = vfmaq_f16(y, vld1q_f16(&w[104]), x104_111);
        y = vfmaq_f16(y, vld1q_f16(&w[112]), x112_120);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (N / 64 >= batch) {
    for (; N - idx >= 64; idx += 64) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      float16x8_t x32_39 = vld1q_f16(&X[idx + 32]);
      float16x8_t x40_47 = vld1q_f16(&X[idx + 40]);
      float16x8_t x48_55 = vld1q_f16(&X[idx + 48]);
      float16x8_t x56_63 = vld1q_f16(&X[idx + 56]);

      if (alpha != 1.0) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
        x32_39 = vmulq_n_f16(x32_39, alpha);
        x40_47 = vmulq_n_f16(x40_47, alpha);
        x48_55 = vmulq_n_f16(x48_55, alpha);
        x56_63 = vmulq_n_f16(x56_63, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        y = vfmaq_f16(y, vld1q_f16(&w[32]), x32_39);
        y = vfmaq_f16(y, vld1q_f16(&w[40]), x40_47);
        y = vfmaq_f16(y, vld1q_f16(&w[48]), x48_55);
        y = vfmaq_f16(y, vld1q_f16(&w[56]), x56_63);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (N / 32 >= batch) {
    for (; N - idx >= 32; idx += 32) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      if (alpha != 1.0) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (N / 16 >= batch) {
    for (; N - idx >= 16; idx += 16) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);

      if (alpha != 1.0) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
      }

      const __fp16 *__restrict w;
      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  for (; N - idx >= 8; idx += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);

    if (alpha != 1.0) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < M; ++j) {
      w = &A[j * N + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t y = vmulq_f16(wvec0_7, x0_7);

      Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
    }
  }
  for (; N - idx >= 4; idx += 4) {
    float32x4_t x0_3 = vcvt_f32_f16(vld1_f16(&X[idx]));

    if (alpha != 1.0) {
      x0_3 = vmulq_n_f32(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < M; ++j) {
      w = &A[j * N + idx];
      float32x4_t wvec0_3 = vcvt_f32_f16(vld1_f16(&w[0]));
      float32x4_t y0 = vmulq_f32(wvec0_3, x0_3);

      Y32[j] += vaddvq_f32(y0);
    }
  }

  // now, N - idx is under 4 : 0 1 2 3 = N - idx
  if (N != idx) {
    float32x4_t x0_3 = vcvt_f32_f16(vld1_f16(&X[idx]));
    for (unsigned int j = N - idx; j < 4; ++j) {
      x0_3[j] = 0;
    }

    if (alpha != 1.0) {
      x0_3 = vmulq_n_f32(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    __fp16 yVal;

    for (unsigned int j = 0; j < M; ++j) {
      w = &A[j * N + idx];
      float32x4_t wvec0_3 = vcvt_f32_f16(vld1_f16(&w[0]));

      for (unsigned int n = N - idx; n < 4; ++n) {
        wvec0_3[n] = 0;
      }

      float32x4_t y0 = vmulq_f32(wvec0_3, x0_3);

      for (unsigned int n = 0; n < N - idx; ++n) {
        Y32[j] += y0[n];
      }
    }
  }

  copy_fp32_to_fp16(M, Y32, Y);
  delete[] Y32;
  return;
}

void hgemv_transpose(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t M,
                     uint32_t N, float alpha, float beta) {
#ifdef OMP_NUM_THREADS
  set_gemv_num_threads(OMP_NUM_THREADS);
#endif
  size_t GEMV_NUM_THREADS = get_gemv_num_threads();

  float *Y32 = new float[N];
  unsigned int idx = 0;
  for (; N - idx >= 8; idx += 8) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7_32 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));

    y0_3_32 = vmulq_n_f32(y0_3_32, beta);
    y4_7_32 = vmulq_n_f32(y4_7_32, beta);

    vst1q_f32(&Y32[idx], y0_3_32);
    vst1q_f32(&Y32[idx + 4], y4_7_32);
  }
  for (; N - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_n_f32(y0_3_32, beta);
    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; N - idx >= 1; idx += 1) {
    Y32[idx] = beta * Y[idx];
  }
  unsigned int i = 0;
  unsigned int N8 = (N >> 3) << 3;
  for (; M - i >= 16; i += 16) {
    __fp16 x[16];
    vst1q_f16(&x[0], vmulq_n_f16(vld1q_f16(&X[i]), alpha));
    vst1q_f16(&x[8], vmulq_n_f16(vld1q_f16(&X[i + 8]), alpha));
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x[0]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * N + idx]), x[1]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 2) * N + idx]), x[2]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 3) * N + idx]), x[3]);

      float16x8_t w2vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 4) * N + idx]), x[4]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 5) * N + idx]), x[5]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 6) * N + idx]), x[6]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 7) * N + idx]), x[7]);

      float16x8_t w3vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 8) * N + idx]), x[8]);
      w3vec0_7_f16 =
        vfmaq_n_f16(w3vec0_7_f16, vld1q_f16(&A[(i + 9) * N + idx]), x[9]);
      w3vec0_7_f16 =
        vfmaq_n_f16(w3vec0_7_f16, vld1q_f16(&A[(i + 10) * N + idx]), x[10]);
      w3vec0_7_f16 =
        vfmaq_n_f16(w3vec0_7_f16, vld1q_f16(&A[(i + 11) * N + idx]), x[11]);

      float16x8_t w4vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 12) * N + idx]), x[12]);
      w4vec0_7_f16 =
        vfmaq_n_f16(w4vec0_7_f16, vld1q_f16(&A[(i + 13) * N + idx]), x[13]);
      w4vec0_7_f16 =
        vfmaq_n_f16(w4vec0_7_f16, vld1q_f16(&A[(i + 14) * N + idx]), x[14]);
      w4vec0_7_f16 =
        vfmaq_n_f16(w4vec0_7_f16, vld1q_f16(&A[(i + 15) * N + idx]), x[15]);

      wvec0_7_f16 = vaddq_f16(wvec0_7_f16, w3vec0_7_f16);
      w2vec0_7_f16 = vaddq_f16(w2vec0_7_f16, w4vec0_7_f16);

      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
      y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }

    if (N != N8) {
      unsigned int idx = N8;

      float y0_7[8];
      float v0[8], v1[8], v2[8], v3[8];
      float v4[8], v5[8], v6[8], v7[8];
      float v8[8], v9[8], v10[8], v11[8];
      float v12[8], v13[8], v14[8], v15[8];

      unsigned int n = 0;
      for (; n < N - idx; ++n) {
        y0_7[n] = Y32[idx + n];

        v0[n] = A[i * N + idx + n];
        v1[n] = A[(i + 1) * N + idx + n];
        v2[n] = A[(i + 2) * N + idx + n];
        v3[n] = A[(i + 3) * N + idx + n];
        v4[n] = A[(i + 4) * N + idx + n];
        v5[n] = A[(i + 5) * N + idx + n];
        v6[n] = A[(i + 6) * N + idx + n];
        v7[n] = A[(i + 7) * N + idx + n];
        v8[n] = A[(i + 8) * N + idx + n];
        v9[n] = A[(i + 9) * N + idx + n];
        v10[n] = A[(i + 10) * N + idx + n];
        v11[n] = A[(i + 11) * N + idx + n];
        v12[n] = A[(i + 12) * N + idx + n];
        v13[n] = A[(i + 13) * N + idx + n];
        v14[n] = A[(i + 14) * N + idx + n];
        v15[n] = A[(i + 15) * N + idx + n];
      }
      for (; n < 8; ++n) {
        y0_7[n] = 0;
        v0[n] = v1[n] = v2[n] = v3[n] = 0;
        v4[n] = v5[n] = v6[n] = v7[n] = 0;
        v8[n] = v9[n] = v10[n] = v11[n] = 0;
        v12[n] = v13[n] = v14[n] = v15[n] = 0;
      }

      float32x4_t y0_3 = vld1q_f32(&y0_7[0]);
      float32x4_t y4_7 = vld1q_f32(&y0_7[4]);

      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v0[0]), x[0]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v1[0]), x[1]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v2[0]), x[2]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v3[0]), x[3]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v4[0]), x[4]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v5[0]), x[5]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v6[0]), x[6]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v7[0]), x[7]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v8[0]), x[8]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v9[0]), x[9]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v10[0]), x[10]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v11[0]), x[11]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v12[0]), x[12]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v13[0]), x[13]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v14[0]), x[14]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v15[0]), x[15]);

      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v0[4]), x[0]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v1[4]), x[1]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v2[4]), x[2]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v3[4]), x[3]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v4[4]), x[4]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v5[4]), x[5]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v6[4]), x[6]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v7[4]), x[7]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v8[4]), x[8]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v9[4]), x[9]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v10[4]), x[10]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v11[4]), x[11]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v12[4]), x[12]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v13[4]), x[13]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v14[4]), x[14]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v15[4]), x[15]);

      for (unsigned int n = 0; n < N - idx; ++n) {
        if (n < 4)
          Y32[idx + n] = y0_3[n];
        else
          Y32[idx + n] = y4_7[n];
      }
    }
  }
  for (; M - i >= 8; i += 8) {
    __fp16 x[8];
    vst1q_f16(&x[0], vmulq_n_f16(vld1q_f16(&X[i]), alpha));
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x[0]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * N + idx]), x[1]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 2) * N + idx]), x[2]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 3) * N + idx]), x[3]);

      float16x8_t w2vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 4) * N + idx]), x[4]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 5) * N + idx]), x[5]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 6) * N + idx]), x[6]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 7) * N + idx]), x[7]);

      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
      y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }

    if (N != N8) {
      unsigned int idx = N8;

      float y0_7[8];
      float v0[8], v1[8], v2[8], v3[8];
      float v4[8], v5[8], v6[8], v7[8];

      unsigned int n = 0;
      for (; n < N - idx; ++n) {
        y0_7[n] = Y32[idx + n];

        v0[n] = A[i * N + idx + n];
        v1[n] = A[(i + 1) * N + idx + n];
        v2[n] = A[(i + 2) * N + idx + n];
        v3[n] = A[(i + 3) * N + idx + n];
        v4[n] = A[(i + 4) * N + idx + n];
        v5[n] = A[(i + 5) * N + idx + n];
        v6[n] = A[(i + 6) * N + idx + n];
        v7[n] = A[(i + 7) * N + idx + n];
      }
      for (; n < 8; ++n) {
        y0_7[n] = 0;
        v0[n] = v1[n] = v2[n] = v3[n] = 0;
        v4[n] = v5[n] = v6[n] = v7[n] = 0;
      }

      float32x4_t y0_3 = vld1q_f32(&y0_7[0]);
      float32x4_t y4_7 = vld1q_f32(&y0_7[4]);

      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v0[0]), x[0]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v1[0]), x[1]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v2[0]), x[2]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v3[0]), x[3]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v4[0]), x[4]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v5[0]), x[5]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v6[0]), x[6]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v7[0]), x[7]);

      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v0[4]), x[0]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v1[4]), x[1]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v2[4]), x[2]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v3[4]), x[3]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v4[4]), x[4]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v5[4]), x[5]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v6[4]), x[6]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v7[4]), x[7]);

      for (unsigned int n = 0; n < N - idx; ++n) {
        if (n < 4)
          Y32[idx + n] = y0_3[n];
        else
          Y32[idx + n] = y4_7[n];
      }
    }
  }
  for (; M - i >= 4; i += 4) {
    __fp16 x[4];
    vst1_f16(&x[0], vmul_n_f16(vld1_f16(&X[i]), alpha));
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x[0]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * N + idx]), x[1]);
      float16x8_t w2vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 2) * N + idx]), x[2]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 3) * N + idx]), x[3]);

      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
      y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }
    if (N != N8) {
      unsigned int idx = N8;
      float y0_3_0[8];
      float v0[8], v1[8], v2[8], v3[8];
      unsigned int n = 0;
      for (; n < N - idx; ++n) {
        y0_3_0[n] = Y32[idx + n];

        v0[n] = A[i * N + idx + n];
        v1[n] = A[(i + 1) * N + idx + n];
        v2[n] = A[(i + 2) * N + idx + n];
        v3[n] = A[(i + 3) * N + idx + n];
      }
      for (; n < 8; ++n) {
        y0_3_0[n] = 0;
        v0[n] = v1[n] = v2[n] = v3[n] = 0;
      }

      float32x4_t y0_3 = vld1q_f32(&y0_3_0[0]);
      float32x4_t y4_7 = vld1q_f32(&y0_3_0[4]);

      // we can separate mul and accum for faster compute.
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v0[0]), x[0]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v1[0]), x[1]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v2[0]), x[2]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v3[0]), x[3]);

      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v0[4]), x[0]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v1[4]), x[1]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v2[4]), x[2]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v3[4]), x[3]);

      for (unsigned int n = 0; n < N - idx; ++n) {
        if (n < 4)
          Y32[idx + n] = y0_3[n];
        else
          Y32[idx + n] = y4_7[n];
      }
    }
  }
  for (; i < M; ++i) {
    __fp16 x = alpha * (X[i]);
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x);
      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }
    if (N != N8) {
      unsigned int idx = N8;
      float v0[8];
      for (unsigned int j = 0; j < N - idx; ++j) {
        v0[j] = A[i * N + idx + j];
      }
      for (unsigned int j = N - idx; j < 8; ++j) {
        v0[j] = 0;
      }

      for (unsigned int j = 0; j < N - idx; ++j) {
        Y32[idx + j] = Y32[idx + j] + v0[j] * x;
      }
    }
  }
  copy_fp32_to_fp16(N, Y32, Y);
  delete[] Y32;
  return;
}

void haxpy(const unsigned int N, const float alpha, const __fp16 *X,
           __fp16 *Y) {

  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x8_t mulacc = vfmaq_f16(y, v_alphaX8, x);
    vst1q_f16(&Y[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float16x4_t y = vld1_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x4_t mulacc = vfma_f16(y, v_alphaX4, x);
    vst1_f16(&Y[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = Y[idx] + alpha * X[idx];
}

__fp16 hdot(const unsigned int N, const __fp16 *X, const __fp16 *Y) {
  float32x4_t accX0_3 = vmovq_n_f32(0.F);
  float32x4_t accX4_7 = vmovq_n_f32(0.F);

  unsigned int idx = 0;
  unsigned int N8 = (N >> 3) << 3;
  float ret = 0;

  // Adaptive loop for batch size of 8
  for (; idx < N8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    x = vmulq_f16(x, y);
    accX0_3 = vaddq_f32(accX0_3, vcvt_f32_f16(vget_low_f16(x)));
    accX4_7 = vaddq_f32(accX4_7, vcvt_f32_f16(vget_high_f16(x)));
  }
  ret += vaddvq_f32(accX0_3) + vaddvq_f32(accX4_7);

  // Loop for remaining indices
  for (; idx < N; idx++)
    ret += X[idx] * Y[idx];

  return static_cast<__fp16>(ret);
}

__fp16 hnrm2(const unsigned int N, const __fp16 *X) {
  float32x4_t accX0_3 = vmovq_n_f32(0.F);
  float32x4_t accX4_7 = vmovq_n_f32(0.F);

  unsigned int idx = 0;
  unsigned int N8 = (N >> 3) << 3;
  float ret = 0;

  // Adaptive loop for batch size of 8
  for (; idx < N8; idx += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);

    x0_7 = vmulq_f16(x0_7, x0_7);
    accX0_3 = vaddq_f32(accX0_3, vcvt_f32_f16(vget_low_f16(x0_7)));
    accX4_7 = vaddq_f32(accX4_7, vcvt_f32_f16(vget_high_f16(x0_7)));
  }
  ret += vaddvq_f32(accX0_3) + vaddvq_f32(accX4_7);

  // Loop for remaining indices
  for (; idx < N; idx++) {
    ret += X[idx] * X[idx];
  }

  return static_cast<__fp16>(sqrt(ret));
}

void hscal(const unsigned int N, __fp16 *X, const float alpha) {
  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);

    // alpha*X -> X
    float16x8_t mulacc = vmulq_f16(v_alphaX8, x);
    vst1q_f16(&X[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);

    // alpha*X -> X
    float16x4_t mulacc = vmul_f16(v_alphaX4, x);
    vst1_f16(&X[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    X[idx] = alpha * X[idx];
}

float32x4_t vcvtq_f32_u32_bitwise(uint32x4_t u32) {
  constexpr uint32_t offsetValue = 0x4b000000;
  const uint32x4_t offsetInt = vdupq_n_u32(offsetValue);
  return vsubq_f32(vreinterpretq_f32_u32(vorrq_u32(u32, offsetInt)),
                   vreinterpretq_f32_u32(offsetInt));
}

void hcopy(const unsigned int N, const __fp16 *X, __fp16 *Y) {

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t batch = vld1q_f16(&X[idx]);
    vst1q_f16(&Y[idx], batch);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t batch = vld1_f16(&X[idx]);
    vst1_f16(&Y[idx], batch);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = X[idx];
}

void copy_int4_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y) {

  unsigned int idx = 0;

  // keep in mind that : len(X) = N, and len(Y) = 2*N

  // processing batch of 16

  float16x8_t y0, y1, y2, y3;
  float16x4_t yh0, yh1;

  uint8_t low0, low1, high0, high1;

  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);

    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    for (unsigned int i = 0; i < 8; ++i) {
      low0 = low[i] >> 4;
      low1 = low[i] & 0x0f;

      high0 = high[i] >> 4;
      high1 = high[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }

      if (i < 4) {
        y2[2 * i] = high0;
        y2[2 * i + 1] = high1;
      } else {
        y3[2 * (i - 4)] = high0;
        y3[2 * (i - 4) + 1] = high1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
    vst1q_f16(&Y[2 * idx + 16], y2);
    vst1q_f16(&Y[2 * idx + 24], y3);
  }

  // processing remaining batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    for (unsigned int i = 0; i < 8; ++i) {
      low0 = batch[i] >> 4;
      low1 = batch[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
  }

  // pocessing remaining values
  for (; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void copy_int8_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    // convert to u16
    uint16x8_t batch_low_u16 = vmovl_u8(low);
    uint16x8_t batch_high_u16 = vmovl_u8(high);

    // todo : experiment with vcvt_f32_u32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f16(&Y[idx], vcvtq_f16_u16(batch_low_u16));
    vst1q_f16(&Y[idx + 8], vcvtq_f16_u16(batch_high_u16));
  }
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    // convert to u16
    uint16x8_t batch_u16 = vmovl_u8(batch);
    vst1q_f16(&Y[idx], vcvtq_f16_u16(batch_u16));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_int8_to_fp16(const unsigned int N, const int8_t *X, __fp16 *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    int8x16_t batch = vld1q_s8(&X[idx]);
    int8x8_t low = vget_low_s8(batch);
    int8x8_t high = vget_high_s8(batch);

    // convert to s16
    int16x8_t batch_low_s16 = vmovl_s8(low);
    int16x8_t batch_high_s16 = vmovl_s8(high);

    // todo : experiment with vcvt_f32_s32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f16(&Y[idx], vcvtq_f16_s16(batch_low_s16));
    vst1q_f16(&Y[idx + 8], vcvtq_f16_s16(batch_high_s16));
  }
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_s8(&X[idx]);

    // convert to s16
    int16x8_t batch_s16 = vmovl_s8(batch);
    vst1q_f16(&Y[idx], vcvtq_f16_s16(batch_s16));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_fp16_to_fp32(const unsigned int N, const __fp16 *X, float *Y) {
  unsigned int idx = 0;

  for (; N - idx >= 8; idx += 8) {
    float32x4_t y1 = vcvt_f32_f16(vld1_f16(&X[idx]));
    float32x4_t y2 = vcvt_f32_f16(vld1_f16(&X[idx + 4]));

    vst1q_f32(&Y[idx], y1);
    vst1q_f32(&Y[idx + 4], y2);
  }

  for (; N - idx >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float32x4_t y1 = vcvt_f32_f16(x);
    vst1q_f32(&Y[idx], y1);
  }

  for (; idx < N; ++idx) {
    Y[idx] = static_cast<float>(X[idx]);
  }
}

void copy_fp32_to_fp16(const unsigned int N, const float *X, __fp16 *Y) {
  unsigned int idx = 0;

  for (; N - idx >= 8; idx += 8) {
    float32x4_t x1 = vld1q_f32(&X[idx]);
    float32x4_t x2 = vld1q_f32(&X[idx + 4]);

    float16x8_t y1 = vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2));

    vst1q_f16(&Y[idx], y1);
  }

  for (; N - idx >= 4; idx += 4) {
    float32x4_t x1 = vld1q_f32(&X[idx]);

    float16x4_t y1 = vcvt_f16_f32(x1);

    vst1_f16(&Y[idx], y1);
  }

  for (; idx < N; ++idx) {
    Y[idx] = static_cast<__fp16>(X[idx]);
  }
}

unsigned int isamax(const unsigned int N, const __fp16 *X) {

  unsigned int retIdx;
  __fp16 maxNum;

  uint16_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16x8_t stride = vmovq_n_u16(8);
  float16x8_t batch = vld1q_f16(&X[0]);
  uint16x8_t curr_index = vld1q_u16(indices);
  uint16x8_t max_index = curr_index;

  unsigned int idx = 8;

  // processing batch of 8
  // stop before idx reaches UNIT16_MAX
  for (; ((N - idx) >= 8) && ((idx + 8) <= UINT16_MAX); idx += 8) {
    float16x8_t values = vld1q_f16(&X[idx]);
    curr_index = vaddq_u16(curr_index, stride);

    // comparison
    uint16x8_t mask = vcgtq_f16(batch, values);

    // blend values and indices based on the mask
    batch = vbslq_f16(mask, batch, values);
    max_index = vbslq_u16(mask, max_index, curr_index);
  }

  // storing indices and max values
  __fp16 maxVal[8];
  vst1q_f16(maxVal, batch);
  vst1q_u16(indices, max_index);

  // getting the index of the maxima
  maxNum = maxVal[0];
  retIdx = indices[0];
  for (unsigned int i = 1; i < 8; i++) {
    if (maxVal[i] > maxNum) {
      maxNum = maxVal[i];
      retIdx = indices[i];
    }
  }

  // if idx is more than UNIT16_MAX
  if ((N > UINT16_MAX) && (N - idx) >= 4) {
    uint32_t indices_u32[] = {idx, idx + 1, idx + 2, idx + 3};
    uint32x4_t stride_u32 = vmovq_n_u32(4);
    float16x4_t batch_4 = vld1_f16(&X[0]);
    uint32x4_t curr_index_u32 = vld1q_u32(indices_u32);
    uint32x4_t max_index_u32 = curr_index_u32;

    idx += 4;
    // processing batch of 4
    for (; (N - idx) >= 4; idx += 4) {
      float16x4_t values_4 = vld1_f16(&X[idx]);
      curr_index_u32 = vaddq_u32(curr_index_u32, stride_u32);

      // comparison
      uint16x4_t mask_4 = vcgt_f16(batch_4, values_4);

      // converting to u32 mask as required by vbslq_u32
      uint32x4_t mask_4_u32 = vmovl_u16(mask_4);

      // blend values and indices based on the mask
      batch_4 = vbsl_f16(mask_4, batch_4, values_4);
      max_index_u32 = vbslq_u32(mask_4_u32, max_index_u32, curr_index_u32);
    }

    // storing indices and max values
    __fp16 maxVal_4[4];
    vst1_f16(maxVal_4, batch_4);
    vst1q_u32(indices_u32, max_index_u32);

    // getting the index of the maxima
    for (unsigned int i = 0; i < 4; i++) {
      if (maxVal_4[i] > maxNum) {
        maxNum = maxVal_4[i];
        retIdx = indices_u32[i];
      }
    }
  }

  // processing remaining values
  for (; idx < N; idx++) {
    if (X[idx] > maxNum) {
      maxNum = X[idx];
      retIdx = idx;
    }
  }

  return retIdx;
}

void custom_hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                  uint32_t N, uint32_t K, float alpha, float beta, bool TransA,
                  bool TransB) {
  hgemm(A, B, C, M, N, K, alpha, beta, TransA, TransB);
}

void ele_mul(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (alpha != 1.f) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vmulq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = alpha * X[i] * Y[i] + beta * Z[i];
    else
      Z[i] = alpha * X[i] * Y[i];
    ++i;
  }
}

void ele_add(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (alpha != 1.f) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vaddq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] + alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] + alpha * Y[i];
    ++i;
  }
}

void ele_sub(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (alpha != 1.f) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vsubq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] - alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] - alpha * Y[i];
    ++i;
  }
}

void ele_div(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (alpha != 1.f) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vdivq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] / (alpha * Y[i]) + beta * Z[i];
    else
      Z[i] = X[i] / (alpha * Y[i]);
    ++i;
  }
}

void inv_sqrt_inplace(const unsigned int N, __fp16 *X) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t x0_7_sqrt = vsqrtq_f16(x0_7);
    float16x8_t ones = vmovq_n_f16(1);
    float16x8_t x0_7_sqrt_div = vdivq_f16(ones, x0_7_sqrt);
    vst1q_f16(&X[i], x0_7_sqrt_div);
  }
  while (i < N) {
    X[i] = (1 / std::sqrt(static_cast<float>(X[i])));
    ++i;
  }
}

bool isValid(const size_t N, const __fp16 *input) {
  bool temp = 0;
  size_t i = 0;
  __fp16 inf_s = std::numeric_limits<float>::infinity();
  float16x8_t inf = vdupq_n_f16(inf_s);
  uint16x8_t zero = vdupq_n_f16(0);

  for (; N - i >= 8; i += 8) {
    float16x8_t vec = vld1q_f16(&input[i]);

    uint16x8_t vcmp = vceqq_f16(vec, vec);

    vcmp = vceqq_f16(vcmp, zero);

    if (vaddvq_u16(vcmp)) {
      return false;
    }

    vcmp = vceqq_f16(vec, inf);

    if (vaddvq_u16(vcmp)) {
      return false;
    }
  }

  while (i < N) {
    if (input[i] != input[i] ||
        input[i] == std::numeric_limits<float>::infinity()) {
      return false;
    }
    ++i;
  }
  return true;
}

#endif
} // namespace nntrainer::neon
