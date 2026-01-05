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
    const float *Biases;  // for inference only
    const float *Weights; // for inference only

    float *MutableWeights; // for training mode
    float *MutableBiases;  // for training mode

    float *NeuronValues;
    float *Errors;
    uint16_t Size;
    BaseLayer *PreviousLayer;
    BaseLayer *NextLayer;
    ActivationKind activationKind;

    BaseLayer();
    virtual ~BaseLayer();

    virtual void FeedForward(bool inferenceOnly = true) = 0;
    virtual void CalculateGradients(const float *desiredValues) = 0;
    virtual void UpdateWeights(float learningRate) = 0;
    virtual void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) = 0;
    virtual void LoadData(const float *weights, const float *biases) = 0;
};

class DenseLayer : public BaseLayer
{
public:
    DenseLayer(uint16_t size, ActivationKind kind);
    ~DenseLayer();
    void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) override;
    void FeedForward(bool inferenceOnly = true) override;
    void CalculateGradients(const float *desiredValues);
    void UpdateWeights(float learningRate);
    void LoadData(const float *weights, const float *biases);
};

class InputLayer : public BaseLayer
{
public:
    InputLayer(uint16_t size);
    ~InputLayer();
    void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) override;
    void FeedForward(bool inferenceOnly = true) override;
    void CalculateGradients(const float *desiredValues);
    void UpdateWeights(float learningRate);
    void LoadData(const float *weights, const float *biases);
};

class OutputLayer : public BaseLayer
{
public:
    OutputLayer(uint16_t size, ActivationKind kind);
    ~OutputLayer();
    void InitLayer(NeuralNetwork *nn, uint16_t size, BaseLayer *previous, BaseLayer *next, bool inferenceOnly = false) override;
    void FeedForward(bool inferenceOnly = true) override;
    void CalculateGradients(const float *desiredValues);
    void UpdateWeights(float learningRate);
    void LoadData(const float *weights, const float *biases);

private:
    void FeedForward_Softmax(bool inferenceOnly = true);
};

#endif
