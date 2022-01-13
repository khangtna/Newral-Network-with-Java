package com.company.data;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class MLDataset {
    private double[][] inputs;
    private double[][] targets;
    private List<MLData> data;

    public MLDataset() {
        data = new ArrayList<>();
    }

    public MLDataset(double[][] inputs, double[][] targets) {
        this.data = new ArrayList<>();
        this.inputs = inputs;
        this.targets = targets;
        for (int i = 0; i < this.inputs.length; i++) {
            this.data.add(new MLData(inputs[i], targets[i]));
        }
    }

    public void addMLData(MLData mlData) {
        this.data.add(mlData);
    }
}
