#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "layers.h"
#include "layerData.h"

class NeuralNetwork
{
private:
    void TrainSingle(float *input, float *desired, uint16_t inputLength, float learningRate);
    void initTrainingMode();
    void initInferenceMode();
    const LayerData *loadedModelLayers = nullptr;
    uint8_t loadedModelLayerCount = 0;

public:
    BaseLayer **allLayer;
    uint8_t stackingIndex;
    uint8_t totalLayers;
    NeuralNetwork(uint8_t totalLayers);
    ~NeuralNetwork();
    void Train(float *inputs, float *desired, uint16_t totalItems, uint16_t inputItemCount, uint16_t epochs, float learningRate);
    NeuralNetwork &StackLayer(BaseLayer *layer);
    void Build(bool inferenceOnly);
    float *Predict(float *inputs, uint16_t inputLength);
    bool LoadTrainedData(const LayerData *layers, uint8_t layerCount);
};

#endif
