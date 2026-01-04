#include <cstdint>

struct LayerData
{
    float *weights;
    float *bias;
    uint16_t inputSize;
    uint16_t outputSize;
};
