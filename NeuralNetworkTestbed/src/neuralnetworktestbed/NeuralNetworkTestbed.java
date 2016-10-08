package neuralnetworktestbed;
import org.ejml.data.*;
import org.ejml.simple.*;
import java.util.Random;
import javafx.beans.property.SimpleMapProperty;

public class NeuralNetworkTestbed {

    
    public static void main(String[] args) {
        
        //RunSimpleNN(10000);
        //RunTwoLayerNN(1000);
        RunMultiLayerNN(1);
    }
    
    private static void RunSimpleNN(int iterations) {
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
        
        simNN.train(trainingInputSet, trainingAnswers, iterations);
        
        System.out.print("Weights After training: ");
        simNN.PrintSynapseWeights();
        
        System.out.println("Considering new situation [1, 0, 0] -> ???");
        
        double[][] t = new double[][] {{1,0,0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        
        System.out.println(simNN.think(thinkSet));
    }
    
    private static void RunTwoLayerNN(int iterations) {
        double[][] l1 = new double[][] {
            {-0.16595599,  0.44064899, -0.99977125, -0.39533485},
            {-0.70648822, -0.81532281, -0.62747958, -0.30887855},
            {-0.20646505,  0.07763347, -0.16161097,  0.370439  }
        };
        double[][] l2 = new double[][] {
            {-0.5910955 },
            { 0.75623487},
            {-0.94522481},
            { 0.34093502}
        };
        
        NeuralLayer nl1 = new NeuralLayer(8, 3);
        //nl1.synapsWeights = new SimpleMatrix(l1);
        NeuralLayer nl2 = new NeuralLayer(1, 8);
        //nl2.synapsWeights = new SimpleMatrix(l2);
        TwoLayerNN Anna = new TwoLayerNN(nl1, nl2);
        
        System.out.println("Random weights: ");
        Anna.PrintsynapsWeights();
        
        double[][] tS = new double[6][3];
        double[][] tA = new double[tS.length][1];
        for(int i = 0; i < tS.length/2; i++) {
            Random rand = new Random();
            double x = rand.nextDouble();
            double y = rand.nextDouble();
            while(x+y > 1) {
                x = rand.nextDouble();
                y = rand.nextDouble();
            }
            
            tS[i][0] = x;
            tS[i][1] = y;
            tS[i][2] = 1;
            
            tA[i][0] = x+y;
            //System.out.println(i + ": " + tS[i][0] + " AVG " + tS[i][1] + " = " + tA[i][0]);
        }
        
        
        
        for(int i = tS.length/2; i < tS.length; i++) {
            Random rand = new Random();
            double x = rand.nextDouble();
            double y = rand.nextDouble();
            while(x-y < 0) {
                x = rand.nextDouble();
                y = rand.nextDouble();
            }
            tS[i][0] = x;
            tS[i][1] = y;
            tS[i][2] = 0;
            
            tA[i][0] = x-y;
            
            //System.out.println(i + ": " + tS[i][0] + "-" + tS[i][1] + " = " + tA[i][0]);
        }
        
        
        
        SimpleMatrix trainingSet = new SimpleMatrix(tS);

        SimpleMatrix trainingAnswers = new SimpleMatrix(tA);
        //trainingAnswers = trainingAnswers.transpose();
        
        Anna.train(trainingSet, trainingAnswers, iterations);
        
        System.out.println("Weights after training: ");
        Anna.PrintsynapsWeights();
        
        System.out.println("Let Anna think about {1, 1, 0} -> ? ");
        ThinkReturns ret = Anna.think(new SimpleMatrix(new double[][] {{0.5, 0.2 ,0}}));
        System.out.println(ret.matrix2);
        
    }

    private static void RunMultiLayerNN(int iterations) {
        
        MultiLayerNN Anna = new MultiLayerNN(3, iterations);
        double[][] TISet = new double[][] 
        {
            //{0, 0, 1},
            //{1, 1, 1},
            //{1, 0, 1},
            {0, 1, 1},
        };
        double[][] TA = new double[][]
        {
            //{0},
            //{1},
            //{1},
            {0}
        };
        
        SimpleMatrix trainingSet = new SimpleMatrix(TISet);
        SimpleMatrix trainingAnswers = new SimpleMatrix(TA);
        trainingAnswers = trainingAnswers.transpose();
        
        Anna.train(trainingSet, trainingAnswers, iterations);
        
        Anna.think(trainingSet);
        
    }
}
