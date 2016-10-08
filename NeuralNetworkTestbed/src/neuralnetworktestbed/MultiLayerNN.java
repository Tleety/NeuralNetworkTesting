
package neuralnetworktestbed;

import java.util.ArrayList;
import org.ejml.simple.*;
import java.util.Random;
import java.util.Arrays;
import java.util.List;
import java.lang.String;

public class MultiLayerNN {
    
    LayerManager Layers;
    int Inputs, Iterations;
    
    public MultiLayerNN(int inputs, int iterations) {
        Inputs = inputs;
        Iterations = iterations;
        Layers = new LayerManager(Inputs);
        
        Layers.printSize("P:");
        Layers.add(3,0);
        Layers.add(2,1);
        Layers.add(4,2);
        Layers.printSize("After 3 Adds");
        Layers.add(8,1);
        Layers.printSize("Add 8N at 1");
        Layers.remove(1);
        Layers.printSize("Remove Layer1");

    }
    
    public void trains(SimpleMatrix trainingSets, SimpleMatrix trainingAnswers, int iterations) {
        for(int i = 0; i < iterations; i++) {
            
        }
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

class LayerManager {
    List<NeuronLayer> Layers = new ArrayList<NeuronLayer>();
    int Inputs = 0;
    LayerManager(int inputs) {
            Inputs = inputs;
    }
    
    //add a layer at position [pos] with [Neurons] neurons. 
    //The layers at and after pos will get moved back 1 position (closer to inserted values).
    public boolean add(int Neurons, int pos) {
        //TODO: So that the first layer(Closest to inputs) can be removed.
        if(pos < 0 || pos > Layers.size()) {
            return false;
        }
        if(Layers.size() == 0) {
            addFirstLayer();
        }
        NeuronLayer newLayer = new NeuronLayer(Neurons, Layers.get(pos).synapsWeights.numRows());
        Layers.add(pos, newLayer);
        if(pos != 0) {
            NeuronLayer reFittedOldLayer = new NeuronLayer(Layers.get(pos-1).synapsWeights.numRows(), Neurons);
            Layers.get(pos-1).synapsWeights = reFittedOldLayer.synapsWeights;
        }
        
        return true;
    }
    
    public boolean remove(int pos) {
        if(pos < 0 || pos > Layers.size()) {
            return false;
        }
        if(pos == Layers.size()){
            int LowerNeurons, HigherNeurons;
            LowerNeurons = Inputs;
            HigherNeurons = Layers.get(pos-1).synapsWeights.numRows();
            NeuronLayer NewHigher = new NeuronLayer(HigherNeurons, LowerNeurons);
            Layers.remove(pos);
            Layers.get(pos-1).synapsWeights = NewHigher.synapsWeights;
        } else {
            int LowerNeurons, HigherNeurons;
            LowerNeurons = Layers.get(pos+1).synapsWeights.numRows();
            HigherNeurons = Layers.get(pos-1).synapsWeights.numRows();
            NeuronLayer NewHigher = new NeuronLayer(HigherNeurons, LowerNeurons);
            Layers.remove(pos);
            Layers.get(pos-1).synapsWeights = NewHigher.synapsWeights;
        }
        
        return true;
    }
    
    private void addFirstLayer() {
        Layers.add(new NeuronLayer(1, Inputs));
        
    }
    
    public void print() {
        int i = 0;
        for(NeuronLayer nl : Layers) {
            System.out.println(i + ":" + nl.synapsWeights);
            i++;
        }
    }
    
    public void print(int i) {
        System.out.println(Layers.get(i).synapsWeights);
    }
    
    public void print(String s) {
        System.out.println(s);
        int i = 0;
        for(NeuronLayer nl : Layers) {
            System.out.println(i + ":" + nl.synapsWeights);
            i++;
        }
    }
    
    public void print(int i, String s) {
        System.out.println(s);
        System.out.println(Layers.get(i).synapsWeights);
    }
    
    public void printSize() {
        int i = 0;
        for(NeuronLayer nl : Layers) {
            System.out.println(i + ":" + "[" + nl.synapsWeights.numRows() + ", " + nl.synapsWeights.numCols() + "]");
            i++;
        }
    }
    
    public void printSize(int i) {
        System.out.println("[" + Layers.get(i).synapsWeights.numRows() + ", " + Layers.get(i).synapsWeights.numCols() + "]");
    }
    
    public void printSize(String s) {
        System.out.println(s);
        int i = 0;
        for(NeuronLayer nl : Layers) {
            System.out.println(i + ":" + "[" + nl.synapsWeights.numRows() + ", " + nl.synapsWeights.numCols() + "]");
            i++;
        }
    }
    
    public void printSize(int i, String s) {
        System.out.println(s);
        System.out.println("[" + Layers.get(i).synapsWeights.numRows() + ", " + Layers.get(i).synapsWeights.numCols() + "]");
    }
}

class NeuronLayer {
    SimpleMatrix synapsWeights;
    NeuronLayer(int Neurons, int SynapsesFromLastLayer) {
        Random rand = new Random();
        synapsWeights = SimpleMatrix.random(Neurons, SynapsesFromLastLayer, -1, 1, rand);
    }
}