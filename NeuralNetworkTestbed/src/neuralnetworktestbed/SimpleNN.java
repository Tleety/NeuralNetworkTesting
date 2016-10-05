package neuralnetworktestbed;

import org.ejml.simple.*;
import org.ejml.data.*;
import org.ejml.factory.*;

public class SimpleNN {

    SimpleMatrix synapsWeights;
    
    public SimpleNN() {
        double[][] d = new double[][] {
            {-0.16595599},
            { 0.44064899},
            {-0.99977125}
        };
        synapsWeights = new SimpleMatrix(d);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    private SimpleMatrix sigmoid(SimpleMatrix m) {
        for(int i = 0; i < m.numRows(); i++) {
            double x = m.get(i, 1);
            m.set(i, 1, sigmoid(x));
        }
        return m;
    }
    
    private double sigmoidDeriv(double x) {
        return x * (1 - x);
    }
    
    private SimpleMatrix sigmoidDeriv(SimpleMatrix m) {
        for(int i = 0; i < m.numRows(); i++) {
            double x = m.get(i, 1);
            m.set(i, 1, sigmoidDeriv(x));
        }
        return m;
    }
    
    public void train(SimpleMatrix trainingSet, SimpleMatrix trainingAnswers, int iterations) {
        for(int i = 0; i < iterations; i++) {
            for(int f = 0; f < trainingSet.numRows(); f++) {
            SimpleMatrix thisSet = trainingSet.extractVector(true, f);
            double output = think(thisSet);
            
            double error = trainingAnswers.get(f) - output;
            
            SimpleMatrix T = thisSet.transpose();
            double dds = sigmoidDeriv(output);
            error = error * dds;
            
            SimpleMatrix Adjustment = T.scale(error);
            
            synapsWeights = synapsWeights.plus(Adjustment);
            }
        }
    }
    
    public double think(SimpleMatrix inputs) {
        double dotM = inputs.dot(synapsWeights);
        return sigmoid(dotM);
    }
    
    public void PrintSynapseWeights() {
        synapsWeights.print();
    }
}
