package neuralnetworktestbed;
import org.ejml.simple.*;
import java.util.LinkedList;

public class MultiLayeredNN {
    
    LinkedList<NeuronLayer> m_NeuronLayers;
    
    double m_Cutoff = 6;
    
    public MultiLayeredNN() {
        m_NeuronLayers = new LinkedList<NeuronLayer>();
        
        m_NeuronLayers.add(new NeuronLayer(6, 4));
        m_NeuronLayers.add(new NeuronLayer(4, 4));
        m_NeuronLayers.add(new NeuronLayer(4, 3));
        m_NeuronLayers.add(new NeuronLayer(3, 1));

    }
    
    public void RemoveLayer(int i) {
        NeuronLayer nextLayer = null;
        NeuronLayer prevLayer = null;
        
        if(m_NeuronLayers.size() > i + 1) {
            nextLayer = m_NeuronLayers.get(i+1);
        }
        
        if(m_NeuronLayers.size() >= i) {
            prevLayer = m_NeuronLayers.get(i-1);
        }
        
        if(nextLayer != null && prevLayer != null) {
            
            if(nextLayer.m_Rows != prevLayer.m_Columns) {
                System.out.println("Rebalancing layer " + (i + 1) + " from " + nextLayer.m_Rows + "," + nextLayer.m_Columns + " to " + prevLayer.m_Columns + "," + nextLayer.m_Columns);
                nextLayer.Rebalance(prevLayer.m_Columns, nextLayer.m_Columns);
            }
        }
        
        m_NeuronLayers.remove(i);
    }
    
    public void AddLayer(int i) {
        
    }

    private double sigmoid(double x) {
        if(x > m_Cutoff) {
            return 1;
        } else if(x < -m_Cutoff) {
            return 0;
        }
        
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
        if(Math.abs(x) > m_Cutoff) {
            return 0;
        }
        
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
            
         LinkedList<SimpleMatrix> outputs = think(trainingSet);
         SimpleMatrix lastLayerOutput = outputs.getLast();
         SimpleMatrix lastLayerError = trainingAnswers.minus(lastLayerOutput);
         SimpleMatrix lastLayerDelta = lastLayerError.elementMult(sigmoidDeriv(lastLayerOutput));
         
         
         for(int n = outputs.size() - 2; n >= 0; n--) {
            SimpleMatrix thisLayerOutput = outputs.get(n);

            SimpleMatrix thisLayerError = lastLayerDelta.mult(m_NeuronLayers.get(n+1).m_SynapsWeights.transpose());
            SimpleMatrix thisLayerDelta = thisLayerError.elementMult(sigmoidDeriv(thisLayerOutput));

            SimpleMatrix thisLayerAdjustment;
            if(n == 0) {
               thisLayerAdjustment = trainingSet.transpose().mult(thisLayerDelta);
            } else {
               thisLayerAdjustment = outputs.get(n-1).transpose().mult(thisLayerDelta);

            }

            SimpleMatrix lastLayerAdjustment = thisLayerOutput.transpose().mult(lastLayerDelta);

            m_NeuronLayers.get(n).m_SynapsWeights = m_NeuronLayers.get(n).m_SynapsWeights.plus(thisLayerAdjustment);
            m_NeuronLayers.get(n+1).m_SynapsWeights = m_NeuronLayers.get(n+1).m_SynapsWeights.plus(lastLayerAdjustment);


            lastLayerOutput = thisLayerOutput.copy();
            lastLayerError = thisLayerError.copy();
            lastLayerDelta = thisLayerDelta.copy();

            }   
        }
    }
    
    public LinkedList<SimpleMatrix> think(SimpleMatrix inputs) { 
        
        LinkedList<SimpleMatrix> outputs = new LinkedList<SimpleMatrix>();
        
        if(outputs.size() == 0 && m_NeuronLayers.size() >= 1) {
            SimpleMatrix output = sigmoid(inputs.mult(m_NeuronLayers.get(0).m_SynapsWeights));
            outputs.addLast(output);
        } else {
            return outputs;
        }
        
        for(int i = 1; i < m_NeuronLayers.size(); i++) {
            SimpleMatrix output = sigmoid(outputs.get(i-1).mult(m_NeuronLayers.get(i).m_SynapsWeights));
            outputs.addLast(output);
        }
        
        return outputs;
    }
    
    public void PrintSynapseWeights() {
      for(int i = 0; i < m_NeuronLayers.size(); i++) {
          m_NeuronLayers.get(i).m_SynapsWeights.print();
      }
    }
}


