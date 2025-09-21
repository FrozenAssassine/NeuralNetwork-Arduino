#ifndef LAYERS_H
#define LAYERS_H

enum ActivationKind {
  Sigmoid,
  Relu,
  Softmax,
  TanH,
  LeakyRelu,
};

float Activation(float x);
float ActivationDeriv(float x);

void FillRandom(float* array, int size);

class BaseLayer {
public:
  float* Biases;
  float* NeuronValues;
  float* Errors;
  float* Weights;
  int Size;
  BaseLayer* PreviousLayer;
  BaseLayer* NextLayer;
  ActivationKind activationKind;

  BaseLayer();
  virtual ~BaseLayer();

  virtual void FeedForward() = 0;
  virtual void Train(const float* desiredValues, float learningRate) = 0;
  virtual void InitLayer(int size, BaseLayer* previous, BaseLayer* next) = 0;
};


class DenseLayer : public BaseLayer {
public:
  DenseLayer(int size, ActivationKind kind);
  ~DenseLayer();
  void InitLayer(int size, BaseLayer* previous, BaseLayer* next) override;
  void FeedForward() override;
  void Train(const float* desiredValues, float learningRate) override;
};


class InputLayer : public BaseLayer {
public:
  InputLayer(int size);
  ~InputLayer();
  void InitLayer(int size, BaseLayer* previous, BaseLayer* next) override;
  void FeedForward() override;
  void Train(const float* desiredValues, float learningRate) override;
};


class OutputLayer : public BaseLayer {
public:
  OutputLayer(int size, ActivationKind kind);
  ~OutputLayer();
  void InitLayer(int size, BaseLayer* previous, BaseLayer* next) override;
  void FeedForward() override;
  void Train(const float* desiredValues, float learningRate) override;

private:
  void FeedForward_Softmax();
};

#endif