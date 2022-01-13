package com.company;

import com.company.Activation.ActivationFunction;
import com.company.data.MLDataset;
import com.company.network.NeuralNetwork;



public class Main {


    private static final double[][] XOR_Input = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}

    };

    private static final double[][] XOR_Label = {
            {0},
            {1},
            {1},
            {0}


    };

    public static void main(String[] args) {

        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 10, 1);
        neuralNetwork.init();
        neuralNetwork.setLearningRate(0.01);
        neuralNetwork.setMomentum(0.5);
        neuralNetwork.setActivationFunction(ActivationFunction.SIGMOID);

        MLDataset dataSet = new MLDataset(XOR_Input, XOR_Label);
        neuralNetwork.train(dataSet, 100000);

        //neuralNetwork.predict(10, 1);
        //neuralNetwork.predict(6, 0.3);
        //neuralNetwork.predict(8.5, 0.3);
        neuralNetwork.predict(0, 0);
        neuralNetwork.predict(0.5,0.6);


    }
}