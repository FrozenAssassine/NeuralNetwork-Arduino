#ifndef LAYERS_H
#define LAYERS_H

#include <Arduino.h>

enum ActivationKind
{
    Sigmoid,
    Relu,
    Softmax,
    TanH,
    LeakyRelu,
};

float Activation(float x);
float ActivationDeriv(float x);

void FillRandom(float *array, uint16_t size);

class NeuralNetwork;

class BaseLayer
{
public:
    float *Biases;
    float *NeuronValues;
    float *Errors;
    float *Weights;
    uint16_t Size;
    BaseLayer *PreviousLayer;
    BaseLayer *NextLayer;
    ActivationKind activationKind;

    BaseLayer();
    virtual ~BaseLayer();

    virtual void FeedForward() = 0;
    virtual void CalculateGradients(const float *desiredValues) = 0;
    virtual void UpdateWeights(float learningRate) = 0;
    virtual void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) = 0;
    virtual void LoadData(float *weights, float *biases) = 0;
};

class DenseLayer : public BaseLayer
{
public:
    DenseLayer(uint16_t size, ActivationKind kind);
    ~DenseLayer();
    void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) override;
    void FeedForward() override;
    void CalculateGradients(const float *desiredValues);
    void UpdateWeights(float learningRate);
    void LoadData(float *weights, float *biases);
};

class InputLayer : public BaseLayer
{
public:
    InputLayer(uint16_t size);
    ~InputLayer();
    void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) override;
    void FeedForward() override;
    void CalculateGradients(const float *desiredValues);
    void UpdateWeights(float learningRate);
    void LoadData(float *weights, float *biases);
};

class OutputLayer : public BaseLayer
{
public:
    OutputLayer(uint16_t size, ActivationKind kind);
    ~OutputLayer();
    void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) override;
    void FeedForward() override;
    void CalculateGradients(const float *desiredValues);
    void UpdateWeights(float learningRate);
    void LoadData(float *weights, float *biases);

private:
    void FeedForward_Softmax();
};

#endif