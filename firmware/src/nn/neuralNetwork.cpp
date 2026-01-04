#include "nn/lossCalculator.h"
#include "nn/neuralNetwork.h"
#include "Arduino.h"
#include "nn/nn_trained.h"

NeuralNetwork::NeuralNetwork(uint8_t totalLayers)
{
    this->allLayer = new BaseLayer *[totalLayers];
    this->totalLayers = totalLayers;
    this->stackingIndex = 0;
}

NeuralNetwork::~NeuralNetwork()
{
    for (uint8_t i = 0; i < this->totalLayers; i++)
    {
        delete this->allLayer[i];
    }
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

void NeuralNetwork::initInferenceMode()
{
    // in inference mode, weights and biases are loaded throught nn_trained.h & errors are not needed
    if (nn_total_layers != this->totalLayers)
    {
        Serial.println("Error: Invalid model layer count");
        exit(0);

        return;
    }

    for (uint8_t i = 0; i < nn_total_layers; i++)
    {
        BaseLayer *prev = (i == 0) ? nullptr : allLayer[i - 1];
        BaseLayer *next = (i == this->totalLayers - 1) ? nullptr : allLayer[i + 1];

        if (nn_layers[i].outputSize != allLayer[i]->Size)
        {
            Serial.println("Error: Loaded model data does not fit on this model");
            Serial.printf("Expected layersize %d received %d\n", allLayer[i]->Size, nn_layers[i].outputSize);
            return;
        }

        allLayer[i]->InitLayer(
            this,
            allLayer[i]->Size,
            prev,
            next,
            true);

        allLayer[i]->LoadData(nn_layers[i].weights, nn_layers[i].bias);
    }
}

void NeuralNetwork::initTrainingMode()
{
    // training mode! Init layers and init weights + bias random for training

    for (uint8_t i = 0; i < this->totalLayers; i++)
    {
        BaseLayer *prev = (i == 0) ? nullptr : allLayer[i - 1];
        BaseLayer *next = (i == this->totalLayers - 1) ? nullptr : allLayer[i + 1];

        allLayer[i]->InitLayer(this, allLayer[i]->Size, prev, next, false);
    }
    return;
}

void NeuralNetwork::Build(bool inferenceOnly)
{
    if (inferenceOnly)
        this->initInferenceMode();
    else
        this->initTrainingMode();
}

float *NeuralNetwork::Predict(float *inputs, uint16_t inputLength)
{
    // give the input neurons the input values:
    for (uint16_t j = 0; j < inputLength; j++)
    {
        this->allLayer[0]->NeuronValues[j] = inputs[j];
    }

    // Feed forward input values throught the network:
    for (uint8_t j = 1; j < this->totalLayers; j++)
    {
        this->allLayer[j]->FeedForward();
    }

    return this->allLayer[this->totalLayers - 1]->NeuronValues;
}

void NeuralNetwork::Train(float *inputs, float *desired, uint16_t totalItems, uint16_t inputItemCount, uint16_t epochs, float learningRate)
{
    Serial.println("Begin training");
    LossCalculator lossCalc = LossCalculator(this);

    for (uint16_t epoch = 0; epoch < epochs; epoch++)
    {
        lossCalc.NextEpoch();
        for (uint16_t i = 0; i < totalItems; i++)
        {

            // feed forward the input values:
            for (uint16_t j = 0; j < inputItemCount; j++)
            {
                this->allLayer[0]->NeuronValues[j] = inputs[i * inputItemCount + j];
            }

            for (uint8_t j = 1; j < this->totalLayers; j++)
            {
                this->allLayer[j]->FeedForward();
            }

            // back propagate the model:
            uint16_t outputSize = this->allLayer[this->totalLayers - 1]->Size;

            // calculate the errors for every layer (back propagete)
            for (uint8_t j = totalLayers; j-- > 0;)
            {
                this->allLayer[j]->CalculateGradients(&desired[i * outputSize]);
            }

            // update the weights using the calculated errors
            for (uint8_t j = 0; j < this->totalLayers; j++)
            {
                this->allLayer[j]->UpdateWeights(learningRate);
            }

            lossCalc.Calculate(&desired[i * outputSize]);
        }

        if (epoch % max(epochs / 100, 1) == 0)
        {
            Serial.print("Epoch ");
            Serial.print(epoch);
            lossCalc.PrintLoss();
        }
    }

    Serial.println("Training Done!");
}