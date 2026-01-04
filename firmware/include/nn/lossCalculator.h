#ifndef LOSS_CALCULATOR_H
#define LOSS_CALCULATOR_H

#include <Arduino.h>

class NeuralNetwork;

class LossCalculator
{
public:
    float totalLossValue;
    uint16_t lossCount;
    NeuralNetwork *neuralNetwork;

    LossCalculator(NeuralNetwork *neuralNetwork);

    void Calculate(float *desired);
    void PrintLoss();
    float MakeLoss();
    void NextEpoch();

private:
    float MSE(float *predicted, float *desired, uint16_t outputSize);
};

#endif