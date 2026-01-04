#include <Arduino.h>
#include <nn/layers.h>
#include <nn/neuralNetwork.h>
#include <nn/predictionHelper.h>

void setup()
{
    Serial.begin(115200);
    delay(1000);

    NeuralNetwork *nn = new NeuralNetwork(3);
    nn->StackLayer(new InputLayer(2));
    nn->StackLayer(new DenseLayer(4, ActivationKind::TanH));
    nn->StackLayer(new OutputLayer(2, ActivationKind::Softmax));
    nn->Build(true); // inference only using trained weights

    float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    Serial.println("Predictions:");
    for (uint8_t i = 0; i < 4; i++)
    {
        float *pred = nn->Predict(inputs[i], 2);
        Serial.printf("Input: [%.0f, %.0f] -> Softmax: [%.4f, %.4f] -> Class: %d\n",
                      inputs[i][0], inputs[i][1], pred[0], pred[1], ArgMax(pred, 2));
    }
}

void loop() {}
