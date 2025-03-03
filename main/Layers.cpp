#include "Layers.h"
#include "Arduino.h"


float Activation(float value, ActivationKind activationKind) {
    switch (activationKind) {
        case ActivationKind::Sigmoid:
            return 1.0 / (1.0 + exp(-value));
        case ActivationKind::Relu:
            return value > 0 ? value : 0;
        case ActivationKind::Softmax:
            return exp(value) / (1.0 + exp(value));
        case ActivationKind::TanH:
            return tanh(value);
        case ActivationKind::LeakyRelu:
            return value > 0 ? value : 0.01 * value;
        default: 
          Serial.println("Invalid Activation Index");
          return value;
    }
}

float ActivationDeriv(float value, ActivationKind activationKind) {
    switch (activationKind) {
        case ActivationKind::Sigmoid:
            return value * (1.0 - value);
        case ActivationKind::Relu:
            return value > 0 ? 1.0 : 0.0;
        case ActivationKind::Softmax:
          return value * (1.0-value);
        case ActivationKind::TanH:
            return 1.0 - value * value;
        case ActivationKind::LeakyRelu:
            return value > 0 ? 1.0 : 0.010;
        default:
          Serial.println("Invalid Activation Index");
          return value;
    }
}

void FillRandom(float* array, int size) {
  for (int i = 0; i < size; i++) {
    float rand = (random(0, 32768) / 32768.0f) * 2 - 1;
    array[i] = rand;
  }
}

///BASE LAYER
BaseLayer::BaseLayer()
  : Biases(nullptr), NeuronValues(nullptr), Errors(nullptr), Weights(nullptr),
    Size(0), PreviousLayer(nullptr), NextLayer(nullptr) {}
BaseLayer::~BaseLayer() {
  delete[] Biases;
  delete[] NeuronValues;
  delete[] Errors;
  delete[] Weights;
}

///DENSE LAYER
DenseLayer::DenseLayer(int size, ActivationKind activationKind)
  : BaseLayer() {
  this->Size = size;
  this->activationKind = activationKind;
}
DenseLayer::~DenseLayer() {}
void DenseLayer::InitLayer(int size, BaseLayer* previous, BaseLayer* next) {
  this->Size = size;
  this->Biases = new float[size];
  this->NeuronValues = new float[size];
  this->Errors = new float[size];
  this->Weights = new float[size * previous->Size];
  this->PreviousLayer = previous;
  this->NextLayer = next;

  FillRandom(this->Biases, size);
  FillRandom(this->Weights, size * previous->Size);
}
void DenseLayer::FeedForward() {
  for (int idx = 0; idx < this->Size; idx++) {
    float sum = 0.0f;
    int index = idx * this->PreviousLayer->Size;
    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      sum += this->PreviousLayer->NeuronValues[j] * this->Weights[index + j];
    }
    this->NeuronValues[idx] = Activation(sum + this->Biases[idx], this->activationKind);
  }
}
void DenseLayer::Train(const float* desiredValues, float learningRate) {
  for (int idx = 0; idx < this->Size; idx++) {
    float err = 0.0f;
    int index = idx * this->PreviousLayer->Size;

    for (int j = 0; j < this->NextLayer->Size; j++) {
      err += (this->NextLayer->Errors[j] * this->NextLayer->Weights[j * this->Size + idx]);
    }
    float error = err * ActivationDeriv(this->NeuronValues[idx], this->activationKind);
    this->Errors[idx] = error;

    error *= learningRate;

    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      this->Weights[index + j] += error * this->PreviousLayer->NeuronValues[j];
    }

    this->Biases[idx] += error;
  }
}

//INPUT LAYER
InputLayer::InputLayer(int size)
  : BaseLayer() {
  this->Size = size;
  this->activationKind = activationKind;
}
InputLayer::~InputLayer() {}
void InputLayer::InitLayer(int size, BaseLayer* previous, BaseLayer* next) {
  this->Size = size;
  this->NeuronValues = new float[size];
  this->PreviousLayer = previous;
  this->NextLayer = next;
}
void InputLayer::FeedForward() {}
void InputLayer::Train(const float* desiredValues, float learningRate) {}


//OUTPUT LAYER
OutputLayer::OutputLayer(int size, ActivationKind activationKind)
  : BaseLayer() {
  this->Size = size;
  this->activationKind = activationKind;
}
OutputLayer::~OutputLayer() {}
void OutputLayer::InitLayer(int size, BaseLayer* previous, BaseLayer* next) {
  this->Size = size;
  this->Biases = new float[size];
  this->NeuronValues = new float[size];
  this->Errors = new float[size];
  this->Weights = new float[size * previous->Size];
  this->PreviousLayer = previous;
  this->NextLayer = next;

  FillRandom(this->Biases, size);
  FillRandom(this->Weights, size * previous->Size);
}

void OutputLayer::FeedForward() {
  for (int idx = 0; idx < this->Size; idx++) {
    float sum = 0.0f;
    int weightIndex = idx * this->PreviousLayer->Size;
    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      sum += this->PreviousLayer->NeuronValues[j] * this->Weights[weightIndex + j];
    }
    this->NeuronValues[idx] = Activation(sum + this->Biases[idx], this->activationKind);
  }
}

void OutputLayer::Train(const float* desiredValues, float learningRate) {
  for (int idx = 0; idx < this->Size; idx++) {
    this->Errors[idx] = desiredValues[idx] - this->NeuronValues[idx];
  }

  for (int idx = 0; idx < this->Size; idx++) {
    float derivNeuronVal = learningRate * this->Errors[idx] * ActivationDeriv(this->NeuronValues[idx], this->activationKind);
    int weightIndex = idx * this->PreviousLayer->Size;

    for (int j = 0; j < this->PreviousLayer->Size; j++) {
      this->Weights[weightIndex + j] += derivNeuronVal * this->PreviousLayer->NeuronValues[j];
    }

    this->Biases[idx] += learningRate * this->Errors[idx] * ActivationDeriv(this->NeuronValues[idx], this->activationKind);
  }
}
