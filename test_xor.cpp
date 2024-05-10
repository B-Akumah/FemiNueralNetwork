/*
 * test :
 *      a simple implementation of neural network on xor example
 *
 */

#include <iostream>
#include "matrix.hpp"
#include "nueral_network.hpp"
#include <vector>
#include <cstdio>


int main() {
    // creating neural network
    // 2 input neurons, 3 hidden neurons and 1 output neuron
    std::vector<uint32_t> topology = {1, 3, 1};
    sp::SimpleNeuralNetwork nn(topology, 0.1);

    //sample dataset
    std::vector<std::vector<float>> trainingInputs = {
            {1},
            {5},
            {4},
            {8},
            {3},

    };
    std::vector<std::vector<float>> trainingOutput = {
            {2},
            {10},
            {8},
            {16},
            {6}
    };

    uint32_t epoch = 100000;


    //training the neural network with randomized data

    /** TRAINING THE MODEL */
    std::cout << "training start\n";

    for (int i = 0; i < epoch; i++) {
        nn.feedForword(trainingInputs[i % 3]);
        nn.backPropagate(trainingOutput[i % 3]);
    }

    std::cout << "training complete\n";

/**USING EXISTING MODEL*/
//
//    //1: Retrieve model from DB to DTO
//    std::vector<uint32_t> imported_topology;
//    std::vector<sp::Matrix2D<float>> imported_weightMatrices;
//    std::vector<sp::Matrix2D<float>> imported_valueMatrices;
//    std::vector<sp::Matrix2D<float>> imported_biasMatrices;
//    float imported_learningRate;
//
//    //2: Convert DTO to SimpleNeuralNetwork
//    sp::SimpleNeuralNetwork imported_nn(
//            imported_topology,
//            imported_weightMatrices,
//            imported_valueMatrices,
//            imported_biasMatrices,
//            imported_learningRate
//    );


    //testing the neural network
    for (std::vector<float> input: trainingInputs) {
        nn.feedForword(input);
        std::vector<float> preds = nn.getPredictions();
        std::cout << input[0] << "  =>  " << preds[0] << "\n" << std::endl;
    }

    return 0;
}