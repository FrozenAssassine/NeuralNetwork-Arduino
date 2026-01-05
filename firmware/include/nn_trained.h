#pragma once

#include "nn/layerData.h"

const float layer0_weights[8] = {1.430434f, -0.973696f, 2.027066f, 1.679277f, 1.048579f, 1.684631f, -0.269726f, 0.690415f};
const float layer0_bias[4] = {-1.282252f, -2.940798f, -0.513053f, -0.578465f};

const float layer1_weights[8] = {-1.153850f, 1.595637f, -1.419836f, -0.087913f, 1.227274f, -1.626652f, 1.463847f, 1.003036f};
const float layer1_bias[2] = {0.497864f, -0.567415f};

const LayerData nn_layers[] = {
    {nullptr, nullptr, 0, 2},
    {layer0_weights, layer0_bias, 2, 4},
    {layer1_weights, layer1_bias, 4, 2}};

const uint8_t nn_total_layers = 3;
