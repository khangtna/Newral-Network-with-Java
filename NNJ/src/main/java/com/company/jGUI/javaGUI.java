package com.company.Activation;

import java.awt.event.ActionListener;

import com.company.Activation.ActivationFunction;
import com.company.data.MLDataset;
import com.company.network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Arrays;

public class javaGUI implements ActionListener {

    private static final double[][] XOR_Input = {
            {5, 2},
            {6, 0.3},
            {7, 0.15},
            {8, 0.1},
            {9, 0.5},
            {10, 1}

    };

    private static final double[][] XOR_Label = {
            {1},
            {0},
            {0},
            {0},
            {1},
            {1}

    };

    double num1 = 0, num2 = 0, num3 = 0;
    double[] result;

    JFrame frame;
    JTextField salaryTextField;
    JTextField timeTextField;
    JTextField rateTextField;
    JTextField preTextField;
    JButton preButton;
    JPanel panel;
    JTextArea textArea;

    javaGUI() {

        frame = new JFrame("DỰ ĐOÁN KHẢ NĂNG CHO VAY");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(780, 450);
        frame.getContentPane().setBackground(Color.lightGray);
        frame.setLayout(null);
        frame.setResizable(false);

        // salary
        JLabel salaryLabel = new JLabel("LƯƠNG: ");
        salaryLabel.setBounds(80, 100, 150, 50);
        salaryLabel.setFont(new Font(null, Font.BOLD, 22));

        salaryTextField = new JTextField();
        salaryTextField.setToolTipText("Triệu");
        salaryTextField.setBounds(230, 100, 200, 40);
        salaryTextField.setFont(new Font("", Font.BOLD, 20));
        salaryTextField.setBackground(Color.white);
        // textfield.setEditable(false);

        // time
        JLabel timeLabel = new JLabel("THỜI GIAN: ");
        timeLabel.setBounds(80, 170, 150, 50);
        timeLabel.setFont(new Font(null, Font.BOLD, 22));

        timeTextField = new JTextField();
        timeTextField.setToolTipText("Năm");
        timeTextField.setBounds(230, 175, 200, 40);
        timeTextField.setFont(new Font("", Font.BOLD, 20));
        timeTextField.setBackground(Color.white);
        // textfield2.setEditable(false);

        // rate
        JLabel rateLabel = new JLabel("TỈ LỆ: ");
        rateLabel.setBounds(80, 245, 150, 50);
        rateLabel.setFont(new Font(null, Font.BOLD, 22));

        rateTextField = new JTextField();
        rateTextField.setBounds(230, 255, 200, 30);
        rateTextField.setFont(new Font("", Font.BOLD, 18));
        //rateTextField.setBackground(Color.white);
        rateTextField.setEditable(false);

        // predict
        preButton = new JButton();
        preButton.setBounds(520, 115, 140, 50);
        preButton.setText("VAY");
        preButton.setFont(new Font("", Font.BOLD, 15));
        preButton.addActionListener(this);
        preButton.setFocusable(false);

        preTextField = new JTextField();
        preTextField.setBounds(490, 185, 200, 50);
        preTextField.setFont(new Font("", Font.BOLD, 15));
        preTextField.setEditable(false);
        preTextField.setHorizontalAlignment(JTextField.CENTER);
        // textfield4.setBorder(BorderFactory.createLineBorder(Color.gray));
        preTextField.setBackground(Color.black);

        // textArea = new JTextArea();
        // textArea.setBounds(100, 270, 200, 50);
        // textArea.setFont(new Font("", Font.BOLD, 15));
        // textArea.setEditable(false);
        // textArea.setBackground(Color.BLACK);

        // frame.add(textArea);
        // frame.add(panel);
        frame.add(preButton);
        frame.add(salaryTextField);
        frame.add(timeTextField);
        frame.add(rateTextField);
        frame.add(preTextField);
        frame.add(salaryLabel);
        frame.add(timeLabel);
        frame.add(rateLabel);

        frame.setVisible(true);

    }

    public void actionPerformed(ActionEvent e) {

        if (e.getSource() == preButton) {
            // lấy value của 2 textfield
            num1 = Double.parseDouble(salaryTextField.getText());
            num2 = Double.parseDouble(timeTextField.getText());

            // build model
            NeuralNetwork neuralNetwork = new NeuralNetwork(2, 10, 1);
            neuralNetwork.init();
            neuralNetwork.setLearningRate(0.01);
            neuralNetwork.setMomentum(0.5);
            neuralNetwork.setActivationFunction(ActivationFunction.SIGMOID);

            MLDataset dataSet = new MLDataset(XOR_Input, XOR_Label);
            neuralNetwork.train(dataSet, 100000);

            // predict
            result = neuralNetwork.predict(num1, num2);

            for (int i = 0; i < result.length; i++) {
                num3 = result[i];
                num3 = (double) Math.round(num3 * 1000) / 1000;
            }

            // gán value cho textfield
            rateTextField.setText(String.valueOf(num3) + " %");

            if (num3 > 0.5) {
                preTextField.setText("BẠN CÓ THỂ VAY!");
                preTextField.setForeground(Color.green);
            } else {
                preTextField.setText("BẠN KHÔNG THỂ VAY!");
                preTextField.setForeground(Color.red);
            }

        }
    }
    public static void main(String[] args) {
        javaGUI jg = new javaGUI();

    }
}
