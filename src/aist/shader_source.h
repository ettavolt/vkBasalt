#pragma once

#include <cstdint>

namespace vkBasalt::aist::shaderSource {
    const uint32_t to_image[] = {
#include "aist/to_image.comp.h"
    };
    const uint32_t from_image[] = {
#include "aist/from_image.comp.h"
    };
}