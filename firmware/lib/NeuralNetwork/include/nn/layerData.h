#pragma once

#include <cstdint>

struct LayerData
{
    const float *weights;
    const float *bias;
    uint16_t inputSize;
    uint16_t outputSize;
};
