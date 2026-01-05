<!--<p align="center">
    <img src="path_to_your_logo" height="150px" width="auto" alt="Neural Network Logo">
</p>
-->

<h1 align="center">Neural Network for ESP32 and Arduino</h1>
<div align="center">
    <img src="https://img.shields.io/github/stars/FrozenAssassine/NeuralNetworkArduino?style=flat"/>
    <img src="https://img.shields.io/github/issues-pr/FrozenAssassine/NeuralNetworkArduino?style=flat"/>
    <img src="https://img.shields.io/github/repo-size/FrozenAssassine/NeuralNetworkArduino?style=flat"/>
</div>

## 🤔 What is this project?

This project is a lightweight neural network implementation designed to run on microcontrollers like the **ESP32** and **Arduino**. It demonstrates how even resource-constrained devices can train and perform simple tasks like **XOR** prediction. Maybe you’ll find a use case for simple robot or sensor projects.

The project has two supported modes, inference and training mode. Inference mode uses an existing torch model and converts it to a header file, which can be loaded to your esp or arduino.
For fun or testing purposes, you can also run your training directly on the microchip, but for larger models, the performance gets weak pretty fast and you run into memory constraints.

## 📎 [Blog to this project](https://medium.com/@FrozenAssassine/neural-network-from-scratch-on-esp32-2a53a7b65f9f)

## 🛠️ Features

- **Inference only**: Use a python script to convert your pytorch models to include file for esp32 and Arduino.
- **On-device training**: Train your neural network directly on ESP32 or Arduino (no weight saving atm).
- **Activation Functions**: Use activation functions like Softmax, Sigmoid, Relu, TanH and LeakyRelu
- **Xavier Initialization**: Optimizes weight distribution for faster training.
- **Simple building structure**: The oop approach makes building the initial model really simple.

## 🔮 Future features

- Save and load weights from on device training
- More layer types

## 🫶 Code considerations

I tried to keep the code as simple and easy to understand as possible. The neural network is completely built using OOP principles, which means that everything is its own class. This is useful for structuring the model later.
For the individual layers, I used the basic principle of inheritance, where there is a BaseLayer class and each layer inherits from it. The BaseLayer also implements some functions, for Training and FeedForward, as well as pointers to the weights, values, biases, and errors. In the inherited classes, those functions can be overriden with the actual training logic and variable implementations. This is very useful for adding new layers.

## 🏗️ Run the code

1. Clone this repository and open the project with PlatformIO.
2. Upload the code to your ESP32 or Arduino
3. Monitor the predictions via Serial Monitor at 115200 baud rate.

## 1. Training mode

```cpp
#include "nn/layers.h"
#include "nn/neuralNetwork.h"
#include <nn/predictionHelper.h>
#include <Arduino.h>

void TrainAndTest()
{
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

  nn->Train((float *)inputs, (float *)desired, 4, 220, 0.1);

  Serial.println("Predictions:");
  for (uint8_t i = 0; i < 4; i++)
  {
    float *pred = nn->Predict(inputs[i]);
    Serial.printf(
        "Input: [%.0f, %.0f] -> Softmax: [%.4f, %.4f] -> Class: %d\n",
        inputs[i][0], inputs[i][1], pred[0], pred[1], ArgMax(pred, 2));
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  TrainAndTest();
}
void loop() { }
```

**Output:**

```
Training Done!
Predictions:
Input: [0, 0] -> Softmax: [0.9665, 0.0335] -> Class: 0
Input: [0, 1] -> Softmax: [0.0324, 0.9676] -> Class: 1
Input: [1, 0] -> Softmax: [0.0783, 0.9217] -> Class: 1
Input: [1, 1] -> Softmax: [0.9355, 0.0645] -> Class: 0
```

## 2. Inference only

```cpp
#include "nn/layers.h"
#include "nn/neuralNetwork.h"
#include <nn/predictionHelper.h>
#include <Arduino.h>
#include "nn_trained.h" //your generated header file with weights and biases

void InferenceOnly()
{
  Serial.println("Testing model inference only (XOR Classification)");

  NeuralNetwork *nn = new NeuralNetwork(3);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(4, ActivationKind::TanH));
  nn->StackLayer(new OutputLayer(2, ActivationKind::Softmax));

  //load your weights and biases
  nn->LoadTrainedData(nn_layers, nn_total_layers);

  nn->Build(true); // inference only

  float inputs[4][2] = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}};

  Serial.println("Predictions:");
  for (uint8_t i = 0; i < 4; i++)
  {
    float *pred = nn->Predict(inputs[i]);
    Serial.printf(
        "Input: [%.0f, %.0f] -> Softmax: [%.4f, %.4f] -> Class: %d\n",
        inputs[i][0], inputs[i][1], pred[0], pred[1], ArgMax(pred, 2));
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  InferenceOnly();
}
void loop() { }
```

**Output:**

```
Testing model inference only (XOR Classification)
Predictions:
Input: [0, 0] -> Softmax: [0.9523, 0.0477] -> Class: 0
Input: [0, 1] -> Softmax: [0.0702, 0.9298] -> Class: 1
Input: [1, 0] -> Softmax: [0.0817, 0.9183] -> Class: 1
Input: [1, 1] -> Softmax: [0.9112, 0.0888] -> Class: 0
```
