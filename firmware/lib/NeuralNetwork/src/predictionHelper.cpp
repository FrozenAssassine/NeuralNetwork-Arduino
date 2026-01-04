#include "nn/predictionHelper.h"
#include <Arduino.h>

uint16_t ArgMax(const float *array, uint16_t length)
{
    uint16_t maxIdx = 0;
    float maxVal = array[0];

    for (uint16_t i = 1; i < length; i++)
    {
        if (array[i] > maxVal)
        {
            maxVal = array[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}
