#pragma once

#include "layerData.h"

float layer0_weights[8] = {1.694597f, 1.308419f, -1.314386f, -0.903650f, -1.036660f, 2.091955f, -2.517021f, 2.006923f};
float layer0_bias[4] = {-0.033063f, -0.264372f, 0.208891f, -1.040136f};

float layer1_weights[8] = {-1.449604f, 0.621033f, 1.519490f, -1.582430f, 0.827401f, -0.810482f, -1.936075f, 1.971920f};
float layer1_bias[2] = {-0.410137f, -0.222768f};

const LayerData nn_layers[] = {
    {nullptr, nullptr, 0, 2},
    {layer0_weights, layer0_bias, 2, 4},
    {layer1_weights, layer1_bias, 4, 2}};

const uint8_t nn_total_layers = 3;
