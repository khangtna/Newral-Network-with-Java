package com.company.model;

import com.company.Activation.IActivationFunction;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Getter
@Setter
public class Neuron {

    private UUID neuronId;
    private List<Connection> incomingConnections;
    private List<Connection> outgoingConnections;
    private double bias;
    private double gradient;
    private double output;
    private double outputBeforeActivation;
    private IActivationFunction activationFunction;

    public Neuron() {
        this.neuronId = UUID.randomUUID();
        this.incomingConnections = new ArrayList<>();
        this.outgoingConnections = new ArrayList<>();
        this.bias = 1.0;
    }

    public Neuron(List<Neuron> neurons, IActivationFunction activationFunction) {
        this();
        this.activationFunction = activationFunction;
        for (Neuron neuron : neurons) {
            Connection connection = new Connection(neuron, this);
            neuron.getOutgoingConnections().add(connection);
            this.incomingConnections.add(connection);
        }
    }
    // Feedforward
    public void calculateOutput() {
        this.outputBeforeActivation = 0.0;
        for (Connection connection : incomingConnections) {
            this.outputBeforeActivation += connection.getSynapticWeight() * connection.getFrom().getOutput();
        }
        this.output = activationFunction.output(this.outputBeforeActivation + bias);
    }

    public double loss(double target) {
        return target - output;
    }

    // Backpropagation
    public void calculateGradient(double target) {
        this.gradient = loss(target) * activationFunction.outputDerivative(output);
    }

    public void calculateGradient() {
        this.gradient = outgoingConnections.stream().mapToDouble
                (connection -> connection.getTo().getGradient() * connection.getSynapticWeight()).sum()
                * activationFunction.outputDerivative(output);
    }

    public void updateConnections(double lr, double mu) {
        for (Connection connection : incomingConnections) {
            double prevDelta = connection.getSynapticWeightDelta();
            connection.setSynapticWeightDelta(lr * gradient * connection.getFrom().getOutput());
            connection.updateSynapticWeight(connection.getSynapticWeightDelta() + mu * prevDelta);
        }
    }

}