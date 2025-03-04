#include "LossCalculator.h"
#include "NeuralNetwork.h"
#include "Arduino.h"

NeuralNetwork::NeuralNetwork(int totalLayers)
{
  this->allLayer = new BaseLayer *[totalLayers];
  this->totalLayers = totalLayers;
  this->stackingIndex = 0;
}

NeuralNetwork::~NeuralNetwork()
{
  delete[] this->allLayer;
}

NeuralNetwork &NeuralNetwork::StackLayer(BaseLayer *layer)
{
  if (this->stackingIndex >= this->totalLayers)
  {
    Serial.println("Can not stack any more layers. Check your total layer count.");
    return *this;
  }

  this->allLayer[this->stackingIndex++] = layer;
  return *this;
}

void NeuralNetwork::Build()
{
  for (int i = 0; i < this->totalLayers; i++)
  {
    if (i == 0)
    { // first layer (input)
      allLayer[i]->InitLayer(allLayer[i]->Size, nullptr, allLayer[i + 1]);
    }
    else if (i == this->totalLayers - 1)
    { // output layer
      allLayer[i]->InitLayer(allLayer[i]->Size, allLayer[i - 1], nullptr);
    }
    else
    {
      allLayer[i]->InitLayer(allLayer[i]->Size, allLayer[i - 1], allLayer[i + 1]);
    }
  }
}

float *NeuralNetwork::Predict(float *inputs, int inputLength)
{
  // give the input neurons the input values:
  for (int j = 0; j < inputLength; j++)
  {
    this->allLayer[0]->NeuronValues[j] = inputs[j];
  }

  // Feed forward input values:
  for (int j = 1; j < this->totalLayers; j++)
  {
    this->allLayer[j]->FeedForward();
  }

  return this->allLayer[this->totalLayers - 1]->NeuronValues;
}

void NeuralNetwork::Train(float *inputs, float *desired, int totalItems, int inputItemCount, int epochs, float learningRate)
{
  Serial.println("Begin training");
  LossCalculator lossCalc = LossCalculator(this);

  for (int epoch = 0; epoch < epochs; epoch++)
  {
    lossCalc.NextEpoch();
    for (int i = 0; i < totalItems; i++)
    {

      // feed forward the input values:
      for (int j = 0; j < inputItemCount; j++)
      {
        this->allLayer[0]->NeuronValues[j] = inputs[i * inputItemCount + j];
      }

      for (int j = 1; j < this->totalLayers; j++)
      {
        this->allLayer[j]->FeedForward();
      }

      // back propagate the model:
      for (int j = this->totalLayers - 1; j >= 0; j--)
      {
        this->allLayer[j]->Train(&desired[i], learningRate);
      }
      lossCalc.Calculate(desired);
    }

    if (epoch % (epochs / 100 > 10 ? epochs / 100 : 1) == 0)
    {
      Serial.print("Epoch ");
      Serial.print(epoch);
      lossCalc.PrintLoss();
    }
  }

  Serial.println("Training Done!");
}