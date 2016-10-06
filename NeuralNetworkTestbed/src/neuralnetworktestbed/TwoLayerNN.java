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
            ThinkReturns tr = think(trainingSet);
            
            SimpleMatrix NL1Out = tr.matrix1;
            SimpleMatrix NL2Out = tr.matrix2;

            SimpleMatrix NL2Error = trainingAnswer.minus(NL2Out);
            SimpleMatrix NL2Delta = NL2Error.elementMult(sigmoidDeriv(NL2Out));
            
            SimpleMatrix NL1Error = NL2Delta.mult(NL2.synapsWeights.transpose());
            SimpleMatrix NL1Delta = sigmoidDeriv(NL1Out).elementMult(NL1Error);
            
            SimpleMatrix NL1Adjustment = trainingSet.transpose().mult(NL1Delta);
            SimpleMatrix NL2Adjustment = NL1Out.transpose().mult(NL2Delta);

            NL1.synapsWeights = NL1.synapsWeights.plus(NL1Adjustment);
            NL2.synapsWeights = NL2.synapsWeights.plus(NL2Adjustment);
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
        SimpleMatrix returnMat = m.copy();
        for(int row = 0; row < returnMat.numRows(); row++) {
            for(int col = 0; col < returnMat.numCols(); col++) {
                double x = returnMat.get(row, col);
                returnMat.set(row, col, sigmoid(x));
            }
        }
        return returnMat;
    }
    
    private double sigmoidDeriv(double x) {
        return x * (1 - x);
    }
    
    private SimpleMatrix sigmoidDeriv(SimpleMatrix m) {
        SimpleMatrix returnMat = m.copy();
        for(int row = 0; row < returnMat.numRows(); row++) {
            for(int col = 0; col < returnMat.numCols(); col++) {
                double x = returnMat.get(row, col);
                returnMat.set(row, col, sigmoidDeriv(x));
            }
        }
        return returnMat;
    }
}
