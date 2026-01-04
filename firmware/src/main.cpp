#include "nn/layers.h"
#include "nn/neuralNetwork.h"
#include <Arduino.h>
#include <nn/predictionHelper.h>

void TrainAndTest()
{
  Serial.println("Start Softmax Training (XOR Classification)");

  randomSeed(42); // For reproducible results when debuggin (not required)

  NeuralNetwork *nn = new NeuralNetwork(3);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(4, ActivationKind::TanH));
  nn->StackLayer(new OutputLayer(2, ActivationKind::Softmax));
  nn->Build(false); // training and prediction

  float inputs[4][2] = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}};

  float desired[4][2] = {
      {1, 0},
      {0, 1},
      {0, 1},
      {1, 0}};

  nn->Train((float *)inputs, (float *)desired, 4, 2, 220, 0.1);

  Serial.println("Predictions:");
  for (uint8_t i = 0; i < 4; i++)
  {
    float *pred = nn->Predict(inputs[i], 2);
    Serial.printf(
        "Input: [%.0f, %.0f] -> Softmax: [%.4f, %.4f] -> Class: %d\n",
        inputs[i][0], inputs[i][1], pred[0], pred[1], ArgMax(pred, 2));
  }
}

void InferenceOnly()
{
  Serial.println("Testing model inference only (XOR Classification)");

  NeuralNetwork *nn = new NeuralNetwork(3);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(4, ActivationKind::TanH));
  nn->StackLayer(new OutputLayer(2, ActivationKind::Softmax));
  nn->Build(true); // inference only

  float inputs[4][2] = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}};

  Serial.println("Predictions:");
  for (uint8_t i = 0; i < 4; i++)
  {
    float *pred = nn->Predict(inputs[i], 2);
    Serial.printf(
        "Input: [%.0f, %.0f] -> Softmax: [%.4f, %.4f] -> Class: %d\n",
        inputs[i][0], inputs[i][1], pred[0], pred[1], ArgMax(pred, 2));
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  // InferenceOnly();
  TrainAndTest();
}

void loop()
{
}