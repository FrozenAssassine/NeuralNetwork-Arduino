#include "nn/layers.h"
#include "nn/neuralNetwork.h"
#include "Arduino.h"

float Activation(float value, ActivationKind activationKind)
{
    switch (activationKind)
    {
    case ActivationKind::Sigmoid:
        return 1.0 / (1.0 + exp(-value));
    case ActivationKind::Relu:
        return value > 0 ? value : 0;
    case ActivationKind::TanH:
        return tanh(value);
    case ActivationKind::LeakyRelu:
        return value > 0 ? value : 0.01 * value;
    case ActivationKind::Softmax:
        return 0;
    default:
        Serial.println("Invalid Activation Index");
        return value;
    }
}

float ActivationDeriv(float value, ActivationKind activationKind)
{
    switch (activationKind)
    {
    case ActivationKind::Sigmoid:
        return value * (1.0 - value);
    case ActivationKind::Relu:
        return value > 0 ? 1.0 : 0.0;
    case ActivationKind::TanH:
        return 1.0 - value * value;
    case ActivationKind::LeakyRelu:
        return value > 0 ? 1.0 : 0.010;
    case ActivationKind::Softmax:
        return 0;
    default:
        Serial.println("Invalid Activation Index");
        return value;
    }
}

void ActivationSoftmax(float *values, uint16_t size)
{
    float maxVal = values[0];
    for (uint16_t i = 1; i < size; i++)
    {
        if (values[i] > maxVal)
            maxVal = values[i];
    }

    float sum = 0.0f;
    for (uint16_t i = 0; i < size; i++)
    {
        values[i] = exp(values[i] - maxVal);
        sum += values[i];
    }

    for (uint16_t i = 0; i < size; i++)
    {
        values[i] /= sum;
    }
}

void FillRandom(float *array, uint16_t size)
{
    for (uint16_t i = 0; i < size; i++)
    {
        float rand = (random(0, 32768) / 32768.0) * 2 - 1;
        array[i] = rand;
    }
}

void XavierInitializeWeights(float *weights, uint16_t size, uint16_t fanIn, uint16_t fanOut)
{
    float limit = sqrt(6.0 / (fanIn + fanOut));
    for (uint16_t i = 0; i < size; i++)
    {
        weights[i] = (random(0, 32768) / 32768.0) * (limit * 2) - limit;
    }
}

/// BASE LAYER
BaseLayer::BaseLayer()
    : Biases(nullptr), NeuronValues(nullptr), Errors(nullptr), Weights(nullptr),
      Size(0), PreviousLayer(nullptr), NextLayer(nullptr) {}
BaseLayer::~BaseLayer()
{
    delete[] Biases;
    delete[] NeuronValues;
    delete[] Errors;
    delete[] Weights;
    delete[] MutableWeights;
    delete[] MutableBiases;
}

/// DENSE LAYER
DenseLayer::DenseLayer(uint16_t size, ActivationKind activationKind)
    : BaseLayer()
{
    this->Size = size;
    this->activationKind = activationKind;
}
DenseLayer::~DenseLayer() {}
void DenseLayer::InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly)
{
    this->Size = size;
    this->NeuronValues = new float[size];
    this->PreviousLayer = previous;
    this->NextLayer = next;

    if (!inferenceOnly)
    {
        this->MutableBiases = new float[size];
        this->MutableWeights = new float[size * previous->Size];
        this->Errors = new float[size];

        FillRandom(this->MutableBiases, size);
        XavierInitializeWeights(this->MutableWeights, size * previous->Size, nn->allLayer[0]->Size, nn->allLayer[nn->totalLayers - 1]->Size);
    }
}

void DenseLayer::FeedForward(bool inferenceOnly)
{
    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float sum = 0.0f;
        uint32_t index = idx * this->PreviousLayer->Size;
        for (uint16_t j = 0; j < this->PreviousLayer->Size; j++)
        {
            sum += this->PreviousLayer->NeuronValues[j] * (inferenceOnly ? this->Weights[index + j] : this->MutableWeights[index + j]);
        }

        this->NeuronValues[idx] = Activation(sum + (inferenceOnly ? this->Biases[idx] : this->MutableBiases[idx]), this->activationKind);
    }
}

void DenseLayer::CalculateGradients(const float *desiredValues)
{
    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float err = 0.0f;

        for (uint16_t j = 0; j < this->NextLayer->Size; j++)
        {
            err += (this->NextLayer->Errors[j] * this->NextLayer->MutableWeights[j * this->Size + idx]);
        }

        float error = err * ActivationDeriv(this->NeuronValues[idx], this->activationKind);
        this->Errors[idx] = error;
    }
}

void DenseLayer::UpdateWeights(float learningRate)
{
    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float error = this->Errors[idx] * learningRate;
        uint32_t index = idx * this->PreviousLayer->Size;

        for (uint16_t j = 0; j < this->PreviousLayer->Size; j++)
        {
            this->MutableWeights[index + j] += error * this->PreviousLayer->NeuronValues[j];
        }

        this->MutableBiases[idx] += error;
    }
}

void DenseLayer::LoadData(const float *w, const float *b)
{
    this->Weights = w;
    this->Biases = b;
}

// INPUT LAYER
InputLayer::InputLayer(uint16_t size)
    : BaseLayer()
{
    this->Size = size;
    this->activationKind = activationKind;
}
InputLayer::~InputLayer() {}
void InputLayer::InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly)
{
    this->Size = size;
    this->NeuronValues = new float[size];
    this->PreviousLayer = previous;
    this->NextLayer = next;
}
void InputLayer::FeedForward(bool inferenceOnly) {}
void InputLayer::CalculateGradients(const float *desiredValues) {}
void InputLayer::UpdateWeights(float learningRate) {}
void InputLayer::LoadData(const float *w, const float *b) {}

// OUTPUT LAYER
OutputLayer::OutputLayer(uint16_t size, ActivationKind activationKind)
    : BaseLayer()
{
    this->Size = size;
    this->activationKind = activationKind;
}
OutputLayer::~OutputLayer() {}
void OutputLayer::InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly)
{
    this->Size = size;
    this->NeuronValues = new float[size];
    this->PreviousLayer = previous;
    this->NextLayer = next;

    if (!inferenceOnly)
    {
        this->Errors = new float[size];
        this->MutableBiases = new float[size];
        this->MutableWeights = new float[size * previous->Size];
        XavierInitializeWeights(this->MutableWeights, size * previous->Size, nn->allLayer[0]->Size, nn->allLayer[nn->totalLayers - 1]->Size);
        FillRandom(this->MutableBiases, size);
    }
}

void OutputLayer::FeedForward_Softmax(bool inferenceOnly)
{
    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float sum = 0.0f;
        uint32_t weightIndex = idx * this->PreviousLayer->Size;
        for (uint16_t j = 0; j < this->PreviousLayer->Size; j++)
        {
            sum += this->PreviousLayer->NeuronValues[j] * (inferenceOnly ? this->Weights[weightIndex + j] : this->MutableWeights[weightIndex + j]);
        }

        this->NeuronValues[idx] = sum + (inferenceOnly ? this->Biases[idx] : this->MutableBiases[idx]);
    }
    ActivationSoftmax(this->NeuronValues, this->Size);
}

void OutputLayer::FeedForward(bool inferenceOnly)
{
    if (this->activationKind == ActivationKind::Softmax)
    {
        FeedForward_Softmax(inferenceOnly);
        return;
    }

    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float sum = 0.0f;
        uint32_t weightIndex = idx * this->PreviousLayer->Size;
        for (uint16_t j = 0; j < this->PreviousLayer->Size; j++)
        {
            sum += this->PreviousLayer->NeuronValues[j] * (inferenceOnly ? this->Weights[weightIndex + j] : this->MutableWeights[weightIndex + j]);
        }
        this->NeuronValues[idx] = Activation(sum + (inferenceOnly ? this->Biases[idx] : this->MutableBiases[idx]), this->activationKind);
    }
}

void OutputLayer::CalculateGradients(const float *desiredValues)
{
    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float gradZ;
        float rawError = desiredValues[idx] - this->NeuronValues[idx];

        if (this->activationKind == ActivationKind::Sigmoid || this->activationKind == ActivationKind::Softmax)
        {
            gradZ = rawError;
        }
        else
        {
            gradZ = rawError * ActivationDeriv(this->NeuronValues[idx], this->activationKind);
        }

        this->Errors[idx] = gradZ;
    }
}

void OutputLayer::UpdateWeights(float learningRate)
{
    for (uint16_t idx = 0; idx < this->Size; idx++)
    {
        float gradZ = this->Errors[idx];
        float error = gradZ * learningRate;
        uint32_t weightIndex = idx * this->PreviousLayer->Size;

        for (uint16_t j = 0; j < this->PreviousLayer->Size; j++)
        {
            this->MutableWeights[weightIndex + j] += error * this->PreviousLayer->NeuronValues[j];
        }

        this->MutableBiases[idx] += error;
    }
}

void OutputLayer::LoadData(const float *w, const float *b)
{
    this->Weights = w;
    this->Biases = b;
}
