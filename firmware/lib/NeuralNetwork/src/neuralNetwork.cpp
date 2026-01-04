#include "nn/lossCalculator.h"
#include "nn/neuralNetwork.h"
#include "Arduino.h"

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
    if (this->loadedModelLayers == nullptr || this->loadedModelLayerCount == 0)
    {
        Serial.println("Error: No trained model loaded. Call LoadTrainedData() before Build(true)");
        return;
    }

    if (this->loadedModelLayerCount != this->totalLayers)
    {
        Serial.println("Error: Invalid model layer count");
        return;
    }

    for (uint8_t i = 0; i < this->loadedModelLayerCount; i++)
    {
        BaseLayer *prev = (i == 0) ? nullptr : allLayer[i - 1];
        BaseLayer *next = (i == this->totalLayers - 1) ? nullptr : allLayer[i + 1];

        if (this->loadedModelLayers[i].outputSize != allLayer[i]->Size)
        {
            Serial.println("Error: Loaded model data does not fit on this model");
            Serial.printf("Expected layersize %d received %d\n", allLayer[i]->Size, this->loadedModelLayers[i].outputSize);
            return;
        }

        allLayer[i]->InitLayer(
            this,
            allLayer[i]->Size,
            prev,
            next,
            true);

        allLayer[i]->LoadData(this->loadedModelLayers[i].weights, this->loadedModelLayers[i].bias);
    }
}

void NeuralNetwork::initTrainingMode()
{
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
    for (uint16_t j = 0; j < inputLength; j++)
    {
        this->allLayer[0]->NeuronValues[j] = inputs[j];
    }

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

            for (uint16_t j = 0; j < inputItemCount; j++)
            {
                this->allLayer[0]->NeuronValues[j] = inputs[i * inputItemCount + j];
            }

            for (uint8_t j = 1; j < this->totalLayers; j++)
            {
                this->allLayer[j]->FeedForward();
            }

            uint16_t outputSize = this->allLayer[this->totalLayers - 1]->Size;

            for (uint8_t j = totalLayers; j-- > 0;)
            {
                this->allLayer[j]->CalculateGradients(&desired[i * outputSize]);
            }

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

bool NeuralNetwork::LoadTrainedData(const LayerData *layers, uint8_t layerCount)
{
    if (layers == nullptr || layerCount == 0)
        return false;

    if (layerCount != this->totalLayers)
    {
        Serial.println("Error: Provided model layer count does not match network configuration.");
        return false;
    }

    // Basic validation of layer sizes
    for (uint8_t i = 0; i < layerCount; i++)
    {
        if (layers[i].outputSize != this->allLayer[i]->Size)
        {
            Serial.println("Error: Provided model layer sizes do not match network layers.");
            return false;
        }
    }

    this->loadedModelLayers = layers;
    this->loadedModelLayerCount = layerCount;
    return true;
}