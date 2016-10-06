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
        
        for(int i = 0; i < mc.numRows(); i++) {
            for(int n = 0; n < mc.numCols(); n++) {
                double x = mc.get(i, n);
                mc.set(i, n, sigmoid(x));
            }
        }
        return mc;
    }
    
    private double sigmoidDeriv(double x) {
        return x * (1 - x);
    }
    
    private SimpleMatrix sigmoidDeriv(SimpleMatrix m) {
        SimpleMatrix mc = m.copy();
        
        for(int i = 0; i < mc.numRows(); i++) {
            for(int n = 0; n < mc.numCols(); n++) {
                double x = mc.get(i, n);
                mc.set(i, n, sigmoidDeriv(x));
            }
        }
        return m;
    }
    
    
    public void train(SimpleMatrix trainingSet, SimpleMatrix trainingAnswers, int iterations) {
        for(int i = 0; i < iterations; i++) {
            for(int f = 0; f < trainingSet.numRows(); f++) {
            SimpleMatrix layer0 = trainingSet.extractVector(true, f);           // [3, 1]
            ThinkReturn output = think(layer0);
            
            
            
            SimpleMatrix layer1Output = output.outputLayer1;                    // [4, 1]

            double layer2Output = output.outputLayer2;
            double layer2Error = layer2Output - trainingAnswers.get(f);
            double layer2Delta = layer2Error * sigmoidDeriv(layer2Output);
            
            System.out.println("Output: " + layer2Output);
            
            
            //How much did each L1 value contribute to the L2 error?
            SimpleMatrix layer1Error = m_Layer2.m_SynapsWeights.scale(layer2Delta); //[4, 1] * x = [4, 1]
            
            
            
            //How much should we change?
            SimpleMatrix layer1Delta = layer1Error.elementMult(sigmoidDeriv(layer1Output));     // [4, 1] *' [4, 1] = [4, 1]
            SimpleMatrix layer1Adjustment = layer0.transpose().mult(layer1Delta.transpose());   // [3, 1] * [1, 4] = [3, 4]
            SimpleMatrix layer2Adjustment = layer1Output.scale(layer2Delta);                    // [4, 1] * x = [4, 1]


            m_Layer1.m_SynapsWeights = m_Layer1.m_SynapsWeights.plus(layer1Adjustment);
            m_Layer2.m_SynapsWeights = m_Layer2.m_SynapsWeights.plus(layer2Adjustment);

           // PrintSynapseWeights();
            }
        }
    }
    
    public ThinkReturn think(SimpleMatrix inputs) {
        
        
        SimpleMatrix outputLayer1 = sigmoid(inputs.mult(m_Layer1.m_SynapsWeights));
        double outputLayer2 = sigmoid(inputs.dot(m_Layer2.m_SynapsWeights));
        
        ThinkReturn thinkReturn = new ThinkReturn();
        thinkReturn.outputLayer1 = outputLayer1.transpose();
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
    double outputLayer2;
    
    public ThinkReturn() {
        
    }
}

