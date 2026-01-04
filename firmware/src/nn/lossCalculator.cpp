#include "nn/lossCalculator.h"
#include "nn/neuralNetwork.h"
#include "Arduino.h"

LossCalculator::LossCalculator(NeuralNetwork *neuralNetwork)
{
    this->neuralNetwork = neuralNetwork;
    this->totalLossValue = 0;
}

void LossCalculator::Calculate(float *desired)
{
    this->lossCount++;
    this->totalLossValue += MSE(
        this->neuralNetwork->allLayer[this->neuralNetwork->totalLayers - 1]->NeuronValues,
        desired,
        this->neuralNetwork->allLayer[this->neuralNetwork->totalLayers - 1]->Size);
}
void LossCalculator::PrintLoss()
{
    Serial.print(" loss: ");
    Serial.println(this->MakeLoss());
}
float LossCalculator::MakeLoss()
{
    return totalLossValue / lossCount;
}
void LossCalculator::NextEpoch()
{
    this->lossCount = 0;
    this->totalLossValue = 0.0;
}

float LossCalculator::MSE(float *predicted, float *desired, uint16_t outputSize)
{
    float sum = 0.0;
    for (uint16_t i = 0; i < outputSize; i++)
    {
        float err = predicted[i] - desired[i];
        sum += err * err;
    }
    return sum / outputSize;
}