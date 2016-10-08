
package neuralnetworktestbed;

import java.util.ArrayList;
import org.ejml.simple.*;
import java.util.Random;
import java.util.List;
import java.lang.String;
import java.util.Collections;

public class MultiLayerNN {
    
    WeightsManager Weights;
    int Inputs, Iterations;
    
    public MultiLayerNN(int inputs, int iterations) {
        Inputs = inputs;
        Iterations = iterations;
        Weights = new WeightsManager(Inputs);
        
        Weights.add(2);
        Weights.add(4);
        Weights.add(8);

        Weights.printSize();
    }
    
    //Train Anna with a set of inputs, answers over several iterations.
    public void train(SimpleMatrix trainingSets, SimpleMatrix trainingAnswers, int iterations) {
        for(int i = 0; i < iterations; i++) {
            List<SimpleMatrix> Outputs = think(trainingSets);

            //Backprobagate to calculate the error in each layer.
            SimpleMatrix CLouts = Outputs.get(0);
            SimpleMatrix NLouts = Outputs.get(1);
            SimpleMatrix CLweights = Weights.Layers.get(0).synapsWeights;

            SimpleMatrix CLerror = trainingAnswers.transpose().minus(CLouts);
            SimpleMatrix CLdelta = CLerror.elementMult(sigmoidDeriv(CLouts));
            SimpleMatrix CLadjustment = NLouts.transpose().mult(CLdelta);
            CLweights = CLweights.transpose().plus(CLadjustment);
            
            CLouts = NLouts;
            
            for(int l = 1; l < Outputs.size()-1; l++) {
                //Calc adjustments for the other layers.
                NLouts = Outputs.get(l+1);
                CLerror = CLdelta.mult(CLweights.transpose());
                
                CLweights = Weights.Layers.get(l).synapsWeights;
                
                CLdelta = sigmoidDeriv(CLouts).elementMult(CLerror);
                CLadjustment = NLouts.transpose().mult(CLdelta);
                
                CLweights = CLweights.plus(CLadjustment);
                
                CLouts = NLouts;
            }
            
            NLouts = trainingSets;

            CLerror = CLdelta.mult(CLweights.transpose());
            CLdelta = sigmoidDeriv(CLouts).elementMult(CLerror);
            CLadjustment = NLouts.transpose().mult(CLdelta);
            CLweights = Weights.Layers.get(0).synapsWeights;
            CLweights = CLweights.plus(CLadjustment);
            
        }
    }
    
    public List<SimpleMatrix> think(SimpleMatrix inputs) {
        List<SimpleMatrix> Results = new ArrayList<SimpleMatrix>();
        SimpleMatrix prevOutput = inputs;
        for(int i = Weights.Layers.size()-1; i >= 0; --i) {
            prevOutput = prevOutput.mult(Weights.Layers.get(i).synapsWeights.transpose());
            Results.add(prevOutput);
        }
        Collections.reverse(Results);
        return Results;
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

//Really need to change so that layers are placed and read in the opposite order.
class WeightsManager {
    List<NeuronLayer2> Layers = new ArrayList<NeuronLayer2>();
    int Inputs = 0;
    WeightsManager(int inputs) {
            Inputs = inputs;
    }
    
    //add a layer at position [pos] with [Neurons] neurons. 
    //The layers at and after pos will get moved back 1 position (closer to inserted values).
    public boolean add(int Neurons, int pos) {
        //TODO: So that the layer closest to inputs can be removed.
        if(pos < 0 || pos > Layers.size()) {
            return false;
        }
        if(Layers.size() == 0) {
            addFirstLayer();
        }
        NeuronLayer2 newLayer = new NeuronLayer2(Neurons, Layers.get(pos).synapsWeights.numRows());
        Layers.add(pos, newLayer);
        if(pos != 0) {
            NeuronLayer2 reFittedOldLayer = new NeuronLayer2(Layers.get(pos-1).synapsWeights.numRows(), Neurons);
            Layers.get(pos-1).synapsWeights = reFittedOldLayer.synapsWeights;
        }
        
        return true;
    }
    
    //Add a new leayer before the output.
    public boolean add(int Neurons) {
        return add(Neurons, 0);
    }
    
    public boolean remove(int pos) {
        if(pos < 0 || pos > Layers.size()) {
            return false;
        }
        if(pos == Layers.size()){
            int LowerNeurons, HigherNeurons;
            LowerNeurons = Inputs;
            HigherNeurons = Layers.get(pos-1).synapsWeights.numRows();
            NeuronLayer2 NewHigher = new NeuronLayer2(HigherNeurons, LowerNeurons);
            Layers.remove(pos);
            Layers.get(pos-1).synapsWeights = NewHigher.synapsWeights;
        } else {
            int LowerNeurons, HigherNeurons;
            LowerNeurons = Layers.get(pos+1).synapsWeights.numRows();
            HigherNeurons = Layers.get(pos-1).synapsWeights.numRows();
            NeuronLayer2 NewHigher = new NeuronLayer2(HigherNeurons, LowerNeurons);
            Layers.remove(pos);
            Layers.get(pos-1).synapsWeights = NewHigher.synapsWeights;
        }
        
        return true;
    }
    
    private void addFirstLayer() {
        Layers.add(new NeuronLayer2(1, Inputs));
        
    }
    
    public void print() {
        int i = 0;
        for(NeuronLayer2 nl : Layers) {
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
        for(NeuronLayer2 nl : Layers) {
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
        for(NeuronLayer2 nl : Layers) {
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
        for(NeuronLayer2 nl : Layers) {
            System.out.println(i + ":" + "[" + nl.synapsWeights.numRows() + ", " + nl.synapsWeights.numCols() + "]");
            i++;
        }
    }
    
    public void printSize(int i, String s) {
        System.out.println(s);
        System.out.println("[" + Layers.get(i).synapsWeights.numRows() + ", " + Layers.get(i).synapsWeights.numCols() + "]");
    }
}

class NeuronLayer2 {
    SimpleMatrix synapsWeights;
    NeuronLayer2(int Neurons, int SynapsesFromLastLayer) {
        Random rand = new Random();
        synapsWeights = SimpleMatrix.random(Neurons, SynapsesFromLastLayer, -1, 1, rand);
    }
}
