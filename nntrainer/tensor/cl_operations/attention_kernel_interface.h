// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	blas_kernel_interface.h
 * @date	28 August 2024
 * @brief	Interface for attention OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ATTENTION_KERNEL_INTERFACE_H__
#define __ATTENTION_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Rotary Embedding kernel
 * @param[in] in input tensor
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep maximum timestep
 */
void apply_rotary_emb_cl(Tensor &in, unsigned int dim, unsigned int from,
                         unsigned int max_timestep);

/**
 * @brief attention transform openCL version
 *
 * @param out
 * @param query
 * @param key_cache
 * @param value_cache
 * @param from
 * @param to
 * @param num_head_q
 * @param num_head_kv
 * @param head_dim
 * @return true if successful
 * @return false
 */
bool attentionTransformCl(Tensor &out, Tensor &query, Tensor &key_cache,
                          const Tensor &value_cache, const unsigned int from,
                          const unsigned int to, const unsigned int num_head_q,
                          const unsigned int num_head_kv,
                          unsigned int head_dim);

} // namespace nntrainer
#endif /* __ATTENTION_KERNEL_INTERFACE_H__ */
