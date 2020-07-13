#pragma once

#include <cstdint>

namespace vkBasalt::aist::shaderSource {
    const uint32_t to_image[] = {
#include "aist/to_image.comp.h"
    };
    const uint32_t down_conv_low[] = {
#include "aist/down_conv_low.comp.h"
    };
    const uint32_t shuffle_low[] = {
#include "aist/shuffle_low.comp.h"
    };
    const uint32_t in_2d_sum_low[] = {
#include "aist/in_2d_sum_low.comp.h"
    };
    const uint32_t in_2d_coeff_low[] = {
#include "aist/in_2d_coeff_low.comp.h"
    };
    const uint32_t in_2d_scale_low[] = {
#include "aist/in_2d_scale_low.comp.h"
    };
    const uint32_t up_conv_low[] = {
#include "aist/up_conv_low.comp.h"
    };
    const uint32_t from_image[] = {
#include "aist/from_image.comp.h"
    };
}