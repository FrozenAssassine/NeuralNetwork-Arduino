#ifndef LOSS_CALCULATOR_H
#define LOSS_CALCULATOR_H

class NeuralNetwork;

class LossCalculator{
  public:
    float totalLossValue;
    int lossCount;
    NeuralNetwork *neuralNetwork;

  LossCalculator(NeuralNetwork *neuralNetwork);

    void Calculate(float * desired);
    void PrintLoss();
    float MakeLoss();
    void NextEpoch();

  private:
    float MSE(float * predicted, float *desired, int outputSize);
};

#endif