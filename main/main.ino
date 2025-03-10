#include "Layers.h"
#include "NeuralNetwork.h"

void setup()
{
  Serial.begin(115200);
  delay(1000); //wait for serial console, text might get truncated on pc:

  NeuralNetwork *nn = new NeuralNetwork(3);
  nn->StackLayer(new InputLayer(2));
  nn->StackLayer(new DenseLayer(4, ActivationKind::TanH));
  nn->StackLayer(new OutputLayer(1, ActivationKind::Sigmoid));
  nn->Build();

  float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float desired[4][1] = {{0}, {1}, {1}, {0}};

  nn->Train((float *)inputs, (float *)desired, 4, 2, 600, 0.1);

  // predict stuff:
  for (int i = 0; i < 4; i++)
  {
    float *pred = nn->Predict(inputs[i], 2);
    Serial.print("Prediction: ");
    Serial.print(inputs[i][0]);
    Serial.print(" ");
    Serial.print(inputs[i][1]);
    Serial.print(" = ");
    Serial.println(pred[0]);
  }
}

void loop()
{
  delay(1000);
}