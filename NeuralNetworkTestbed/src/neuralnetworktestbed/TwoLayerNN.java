package neuralnetworktestbed;

import org.ejml.simple.*;

public class TwoLayerNN {
    
    NeuralLayer NL1, NL2;
    
    
    public TwoLayerNN(NeuralLayer layer1, NeuralLayer layer2) {
        NL1 = layer1;
        NL2 = layer2;
    }
    
    public void train(SimpleMatrix trainingSet, SimpleMatrix trainingAnswer, int iterations) {
        for(int i = 0; i < iterations; i++) {
            for(int f = 0; f < trainingSet.numRows(); f++) {
                SimpleMatrix thisSet = trainingSet.extractVector(true, f);
                double thisAnswer = trainingAnswer.get(f);
                
                ThinkReturns tr = think(thisSet);
                SimpleMatrix NL1Out = tr.matrix1;
                SimpleMatrix NL2Out = tr.matrix2;
                
                double NL2Error = thisAnswer - NL2Out.get(0, 0);
                double NL2Delta = NL2Error * sigmoidDeriv(NL2Out.get(0, 0));
                
                SimpleMatrix NL1Error = NL2.synapsWeights.scale(NL2Delta);
                SimpleMatrix mm = sigmoidDeriv(NL1Out);
                SimpleMatrix NL1Delta = NL1Error.transpose().elementMult(mm);
                // Right untill here.
                SimpleMatrix TA = trainingSet.transpose();
                SimpleMatrix NL1Adjustment = TA.mult(NL1Delta);
                SimpleMatrix NL2Adjustment = NL1Out.transpose().scale(NL2Delta);
                
                NL1.synapsWeights.plus(NL1Adjustment);
                NL2.synapsWeights.plus(NL2Adjustment);
            }
        }
    }
    
    public ThinkReturns think(SimpleMatrix inputs) {
        SimpleMatrix e = inputs.mult(NL1.synapsWeights);
        SimpleMatrix NL1Output = sigmoid(e);
        SimpleMatrix NL2Output = sigmoid(NL1Output.mult(NL2.synapsWeights));
        return new ThinkReturns(NL1Output, NL2Output);
    }
    
    public void PrintsynapsWeights() {
        System.out.println(NL1.synapsWeights);
        System.out.println(NL2.synapsWeights);
    }
    
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    private SimpleMatrix sigmoid(SimpleMatrix m) {
        for(int row = 0; row < m.numRows(); row++) {
            for(int col = 0; col < m.numCols(); col++) {
                double x = m.get(row, col);
                m.set(row, col, sigmoid(x));
            }
        }
        return m;
    }
    
    private double sigmoidDeriv(double x) {
        return x * (1 - x);
    }
    
    private SimpleMatrix sigmoidDeriv(SimpleMatrix m) {
        for(int row = 0; row < m.numRows(); row++) {
            for(int col = 0; col < m.numCols(); col++) {
                double x = m.get(row, col);
                m.set(row, col, sigmoidDeriv(x));
            }
        }
        return m;
    }
}
