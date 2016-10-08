package neuralnetworktestbed;
import org.ejml.simple.*;
import org.ejml.data.*;
import org.ejml.factory.*;

public class TwoLayeredNN {
    
    NeuronLayer m_Layer1;
    NeuronLayer m_Layer2;
    
    public TwoLayeredNN() {
        m_Layer1 = new NeuronLayer(4, 3);
        m_Layer2 = new NeuronLayer(1, 4);
    }

 private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    private SimpleMatrix sigmoid(SimpleMatrix m) {
        SimpleMatrix mc = m.copy();
        
        for(int row = 0; row < mc.numRows(); row++) {
            for(int col = 0; col < mc.numCols(); col++) {
                mc.set(row, col, sigmoid(mc.get(row, col)));
            }
        }
        return mc;
    }
    
    private double sigmoidDeriv(double x) {
        return x * (1 - x);
    }
    
    private SimpleMatrix sigmoidDeriv(SimpleMatrix m) {
        SimpleMatrix mc = m.copy();
        
        for(int row = 0; row < mc.numRows(); row++) {
            for(int col = 0; col < mc.numCols(); col++) {
                mc.set(row, col, sigmoidDeriv(mc.get(row, col)));
            }
        }
        return mc;
    }
    
    
    public void train(SimpleMatrix trainingSet, SimpleMatrix trainingAnswers, int iterations) {
        for(int i = 0; i < iterations; i++) {
            
            ThinkReturn output = think(trainingSet);
            
            SimpleMatrix layer1Output = output.outputLayer1;
            SimpleMatrix layer2Output = output.outputLayer2;
            
            SimpleMatrix layer2Error = trainingAnswers.minus(layer2Output);
            SimpleMatrix layer2Delta = layer2Error.elementMult(sigmoidDeriv(layer2Output));
            
            SimpleMatrix layer1Error = layer2Delta.mult(m_Layer2.m_SynapsWeights.transpose());
            SimpleMatrix layer1Delta = layer1Error.elementMult(sigmoidDeriv(layer1Output));
            
            SimpleMatrix layer1Adjustment = trainingSet.transpose().mult(layer1Delta);
            SimpleMatrix layer2Adjustment = layer1Output.transpose().mult(layer2Delta);

            m_Layer1.m_SynapsWeights = m_Layer1.m_SynapsWeights.plus(layer1Adjustment);
            m_Layer2.m_SynapsWeights = m_Layer2.m_SynapsWeights.plus(layer2Adjustment);      
        }
    }
    
    public ThinkReturn think(SimpleMatrix inputs) {      
        SimpleMatrix outputLayer1 = sigmoid(inputs.mult(m_Layer1.m_SynapsWeights));
        SimpleMatrix outputLayer2 = sigmoid(outputLayer1.mult(m_Layer2.m_SynapsWeights));
        
        ThinkReturn thinkReturn = new ThinkReturn();
        thinkReturn.outputLayer1 = outputLayer1;
        thinkReturn.outputLayer2 = outputLayer2;
        
        return thinkReturn;
    }
    
    public void PrintSynapseWeights() {
       m_Layer1.m_SynapsWeights.print();
       m_Layer2.m_SynapsWeights.print();
    }
}

class ThinkReturn {
    SimpleMatrix outputLayer1;
    SimpleMatrix outputLayer2;
}

