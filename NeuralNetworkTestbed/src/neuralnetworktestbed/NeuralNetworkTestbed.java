package neuralnetworktestbed;
import org.ejml.data.*;
import org.ejml.simple.*;
import java.util.Scanner;
import java.text.DecimalFormat;

public class NeuralNetworkTestbed {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        
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
        
        System.out.println("Considering new situation [0.3,0.2,0] -> ???");
        
        double[][] t = new double[][] {{0.3,0.2,0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        
        System.out.println("Answer: " + new DecimalFormat("#0.000000").format(twoLayeredNN.think(thinkSet).outputLayer2.get(0)));
        double x = 1, y = 1, z = 1;
        
                
      while(true) {
          x = in.nextDouble();
          if(x > 0.0 || x > 1.0)
              break;
          
          y = in.nextDouble();
          if(y > 0.0 || y > 1.0)
              break;
          
          z = in.nextDouble();
          if(z > 0.0 || z > 1.0)
              break;
      
          double[][] tx = new double[][] {{x,y,z}};
          SimpleMatrix ts = new SimpleMatrix(tx);
      
          System.out.println("Answer: " + new DecimalFormat("#0.000000").format(twoLayeredNN.think(ts).outputLayer2.get(0)));
      }
    }

}
