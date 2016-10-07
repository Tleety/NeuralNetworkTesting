package neuralnetworktestbed;
import org.ejml.data.*;
import org.ejml.simple.*;
import java.text.DecimalFormat;

import java.util.Random;

public class NeuralNetworkTestbed {

    public static void main(String[] args) {
        
        Random random = new Random();
        
        TwoLayeredNN twoLayeredNN = new TwoLayeredNN();
        
        System.out.print("Weights before training: ");
        twoLayeredNN.PrintSynapseWeights();
        

         double[][] tS = new double[][] {
            //Addition
            {0.0, 0.2, 1},
            {0.1, 0.2, 1},
            {0.2, 0.1, 1},
            {0.3, 0.6, 1},
            {0.4, 0.3, 1},
            {0.5, 0.1, 1},
            {0.6, 0.3, 1},
            {0.7, 0.2, 1},
            {0.8, 0.2, 1},
            {0.9, 0.0, 1},
            //Subtraction
            {0.0, 0.0, 0},
            {0.1, 0.1, 0},
            {0.2, 0.1, 0},
            {0.3, 0.1, 0},
            {0.4, 0.3, 0},
            {0.5, 0.1, 0},
            {0.6, 0.3, 0},
            {0.7, 0.2, 0},
            {0.8, 0.5, 0},
            {0.9, 0.3, 0},
        };
        SimpleMatrix trainingSet = new SimpleMatrix(tS);
 
        double[][] tA = new double[][] {
            { 0.2,  0.3,  0.3,  0.9,  0.7,  0.6,  0.9,  0.9,  1.0,  0.9,  //Addition
              0.0,  0.0,  0.1,  0.1,  0.1,  0.4,  0.3,  0.5,  0.3,  0.6 } //Subtraction
        };
        
        SimpleMatrix trainingInputSet = new SimpleMatrix(tS);
        SimpleMatrix trainingAnswers = new SimpleMatrix(tA);
        
        System.out.println("Training our simple neural network over 10 000 iterations");
        
        twoLayeredNN.train(trainingInputSet, trainingAnswers.transpose(), 60000);
        
        System.out.print("Weights After training: ");
        twoLayeredNN.PrintSynapseWeights();
        double totalError = 0.0;
        int iterations = 1000;
        for(int i = 0; i < iterations; i++) {
            
            double z, x, y;
            do {
            x = random.nextDouble();
            y = random.nextDouble();
            
            z = x - y;
            
            }while(z > 1 || z < 0);
            
        double[][] t = new double[][] {{x,y,0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        
        
        double output = twoLayeredNN.think(thinkSet).outputLayer2.get(0);
        
        totalError += Math.abs(output - z);
        
        System.out.println(x + " + " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        
        System.out.println("avg error: " + new DecimalFormat("#0.000000").format(totalError/iterations));
        
    }

}
