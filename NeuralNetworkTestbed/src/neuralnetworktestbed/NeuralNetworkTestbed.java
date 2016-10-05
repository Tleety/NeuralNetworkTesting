package neuralnetworktestbed;
import org.ejml.data.*;
import org.ejml.simple.*;

public class NeuralNetworkTestbed {

    public static void main(String[] args) {
        SimpleNN simNN = new SimpleNN();
        
        System.out.print("Weights before training: ");
        simNN.PrintSynapseWeights();
        
        double[][] TISet = new double[][] 
        {
            {0, 0, 1},
            {1, 1, 1},
            {1, 0, 1},
            {0, 1, 1},
        };
        double[][] TA = new double[][]
        {
            {0},
            {1},
            {1},
            {0}
        };
        
        SimpleMatrix trainingInputSet = new SimpleMatrix(TISet);
        SimpleMatrix trainingAnswers = new SimpleMatrix(TA);
        
        System.out.println("Training our simple neural network over 10 000 iterations");
        
        simNN.train(trainingInputSet, trainingAnswers, 10000);
        
        System.out.print("Weights After training: ");
        simNN.PrintSynapseWeights();
        
        System.out.println("Considering new situation [1, 0, 0] -> ???");
        
        double[][] t = new double[][] {{1,0,0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        
        System.out.println(simNN.think(thinkSet));
        
    }

}
