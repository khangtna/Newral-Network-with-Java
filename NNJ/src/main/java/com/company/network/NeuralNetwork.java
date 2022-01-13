package com.company.network;


import com.company.Activation.ActivationFunction;
import com.company.Activation.IActivationFunction;
import com.company.Activation.LeakyReLu;
import com.company.Activation.Sigmoid;
import com.company.Activation.Tanh;
import com.company.data.MLData;
import com.company.data.MLDataset;
import com.company.model.Neuron;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NeuralNetwork {

    private static final Logger logger = LogManager.getLogger(NeuralNetwork.class);

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private final List<Neuron> inputLayer;
    private final List<Neuron> hiddenLayer;
    private final List<Neuron> outputLayer;

    private double learningRate = 0.01;
    private double momentum = 0.5;
    private IActivationFunction activationFunction = new Sigmoid(); // default activation function
    private boolean initialized = false;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.inputLayer = new ArrayList<>();
        this.hiddenLayer = new ArrayList<>();
        this.outputLayer = new ArrayList<>();
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        switch (activationFunction) {
            case LEAKY_RELU:
                this.activationFunction = new LeakyReLu();
                break;
            case TANH:
                this.activationFunction = new Tanh();
                break;
            case SIGMOID:
                this.activationFunction = new Sigmoid();
                break;

        }
    }

    public void init() {
        for (int i = 0; i < inputSize; i++) {
            this.inputLayer.add(new Neuron());
        }
        for (int i = 0; i < hiddenSize; i++) {
            this.hiddenLayer.add(new Neuron(this.inputLayer, activationFunction));
        }
        for (int i = 0; i < outputSize; i++) {
            this.outputLayer.add(new Neuron(this.hiddenLayer, activationFunction));
        }
        this.initialized = true;
        logger.info("Network Initialized.");
    }


    public void train(MLDataset set, int epoch) {
        if (!initialized){
            this.init();
        }
        logger.info("Training Starting...");
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(set.getData());

            for (MLData datum : set.getData()) {
                forward(datum.getInputs());
                backward(datum.getTargets());
            }
        }
        logger.info("Training Finished.");
    }

    private void backward(double[] targets) {
        int i = 0;
        for (Neuron neuron : outputLayer) {
            neuron.calculateGradient(targets[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateGradient();
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
        for (Neuron neuron : outputLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
    }

    private void forward(double[] inputs) {
        int i = 0;
        for (Neuron neuron : inputLayer) {
            neuron.setOutput(inputs[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateOutput();
        }
        for (Neuron neuron : outputLayer) {
            neuron.calculateOutput();
        }
    }

    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[outputLayer.size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.get(i).getOutput();
        }
        logger.info("Input : " + Arrays.toString(inputs) + ", Predicted : " + Arrays.toString(output));
        //logger.info(Arrays.toString(output));
        return output;
    }

}