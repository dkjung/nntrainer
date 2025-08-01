/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
@file	bitmat_helpers.h
@brief  bitmap_helpers from tensorflow

==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H_
#include <cstdint>
#include <string>
#include <vector>
namespace tflite {
namespace label_image {
std::vector<uint8_t> read_bmp(const std::string &input_bmp_name, int *width, int *height,
                  int *channels);
} // namespace label_image
} // namespace tflite

#endif // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H
