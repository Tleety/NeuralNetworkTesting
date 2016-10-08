package neuralnetworktestbed;
import org.ejml.data.*;
import org.ejml.simple.*;
import java.text.DecimalFormat;
import java.util.LinkedList;
import java.util.Random;

public class NeuralNetworkTestbed {
    
    static double TwoLayeredAvgError = 0.0;
    static double MultiLayeredAvgError = 0.0;
    
    
    static double MultiLayeredPlusAvgError = 0.0;
    
    public static void main(String[] args) {
      //  System.out.println("Teaching neural network addition and subtraction");
      //  RunTwoLayered();
       // RunMultiLayered();
      //  double improvement = TwoLayeredAvgError/MultiLayeredAvgError * 100 - 100;
       // System.out.println("Multi Layered Improvement: " + new DecimalFormat("#0.00").format(improvement) + "%");
        
        RunMultiLayeredPlus();
    }
    
public static void RunTwoLayered() {
        Random random = new Random(0);
        
        TwoLayeredNN twoLayeredNN = new TwoLayeredNN();
        
        //System.out.print("Weights before training: ");
       // twoLayeredNN.PrintSynapseWeights();
        

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
 
        double[][] tA = new double[][] {
            { 0.2,  0.3,  0.3,  0.9,  0.7,  0.6,  0.9,  0.9,  1.0,  0.9,  //Addition
              0.0,  0.0,  0.1,  0.1,  0.1,  0.4,  0.3,  0.5,  0.3,  0.6 } //Subtraction
        };
        
        SimpleMatrix trainingInputSet = new SimpleMatrix(tS);
        SimpleMatrix trainingAnswers = new SimpleMatrix(tA);
        
      //  System.out.println("Training our simple neural network over 10 000 iterations");
        
        twoLayeredNN.train(trainingInputSet, trainingAnswers.transpose(), 60000);
        
      //  System.out.print("Weights After training: ");
       // twoLayeredNN.PrintSynapseWeights();
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
        
      //  System.out.println(x + " + " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        TwoLayeredAvgError = totalError/iterations;
        System.out.println("Two Layered avg error: " + new DecimalFormat("#0.000000").format(TwoLayeredAvgError));
        
    }

public static void RunMultiLayered() {
    Random random = new Random(0);
        
        MultiLayeredNN multiLayeredNN = new MultiLayeredNN();
        
      //  System.out.print("Weights before training: ");
      //  multiLayeredNN.PrintSynapseWeights();
        

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
 
        double[][] tA = new double[][] {
            { 0.2,  0.3,  0.3,  0.9,  0.7,  0.6,  0.9,  0.9,  1.0,  0.9,  //Addition
              0.0,  0.0,  0.1,  0.1,  0.1,  0.4,  0.3,  0.5,  0.3,  0.6 } //Subtraction
        };
        
        SimpleMatrix trainingInputSet = new SimpleMatrix(tS);
        SimpleMatrix trainingAnswers = new SimpleMatrix(tA);
        
       // System.out.println("Training our simple neural network over 10 000 iterations");
        
        multiLayeredNN.train(trainingInputSet, trainingAnswers.transpose(), 60000);
        
        //System.out.print("Weights After training: ");
        //multiLayeredNN.PrintSynapseWeights();
        double totalError = 0.0;
        int iterations = 1000;
        for(int i = 0; i < iterations; i++) {
            
            double z, x, y;
            do {
            x = random.nextDouble();
            y = random.nextDouble();
            
            z = x + y;
            
            }while(z > 1 || z < 0);
            
        double[][] t = new double[][] {{x,y,1}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        
        
        double output = multiLayeredNN.think(thinkSet).getLast().get(0);
        
        totalError += Math.abs(output - z);
        
       // System.out.println(x + " - " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        MultiLayeredAvgError = totalError/iterations;
        System.out.println("Multi Layered avg error: " + new DecimalFormat("#0.000000").format(MultiLayeredAvgError));
        
}


public static void RunMultiLayeredPlus() {
    Random random = new Random();
        
        MultiLayeredNN multiLayeredNN = new MultiLayeredNN();
        
      //  System.out.print("Weights before training: ");
      //  multiLayeredNN.PrintSynapseWeights();
        

         double[][] tS = new double[][] {
            //Addition
            {0.0, 0.2, 1, 0, 0, 0},
            {0.1, 0.2, 1, 0, 0, 0},
            {0.2, 0.1, 1, 0, 0, 0},
            {0.3, 0.6, 1, 0, 0, 0},
            {0.4, 0.3, 1, 0, 0, 0},
            {0.5, 0.1, 1, 0, 0, 0},
            {0.6, 0.3, 1, 0, 0, 0},
            {0.7, 0.2, 1, 0, 0, 0},
            {0.8, 0.2, 1, 0, 0, 0},
            {0.9, 0.0, 1, 0, 0, 0},
            
            //Subtraction
            {0.0, 0.0, 0, 1, 0, 0},
            {0.1, 0.1, 0, 1, 0, 0},
            {0.2, 0.1, 0, 1, 0, 0},
            {0.3, 0.1, 0, 1, 0, 0},
            {0.4, 0.3, 0, 1, 0, 0},
            {0.5, 0.1, 0, 1, 0, 0},
            {0.6, 0.3, 0, 1, 0, 0},
            {0.7, 0.2, 0, 1, 0, 0},
            {0.8, 0.5, 0, 1, 0, 0},
            {0.9, 0.3, 0, 1, 0, 0},
            
            //Multiplication
            {0.0, 0.2, 0, 0, 1, 0},//0.0
            {0.1, 0.2, 0, 0, 1, 0},//0.02 
            {0.2, 0.1, 0, 0, 1, 0},//0.02
            {0.3, 0.6, 0, 0, 1, 0},//0.18
            {0.4, 0.3, 0, 0, 1, 0},//0.12
            {0.5, 0.1, 0, 0, 1, 0},//0.05
            {0.6, 0.3, 0, 0, 1, 0},//0.18
            {0.7, 0.2, 0, 0, 1, 0},//0.14
            {0.8, 0.2, 0, 0, 1, 0},//0.16
            {0.9, 0.0, 0, 0, 1, 0},//0.0
            //Division
            {0.0, 0.1, 0, 0, 0, 1},//0.0
            {0.1, 0.1, 0, 0, 0, 1},//1.0
            {0.2, 0.5, 0, 0, 0, 1},//0.4
            {0.3, 0.3, 0, 0, 0, 1},//1.0
            {0.4, 0.9, 0, 0, 0, 1},//0.4444
            {0.5, 0.8, 0, 0, 0, 1},//0.6250
            {0.6, 0.7, 0, 0, 0, 1},//0.8571
            {0.3, 0.6, 0, 0, 0, 1},//0.5
            {0.5, 0.6, 0, 0, 0, 1},//0.8333
            {0.9, 1.0, 0, 0, 0, 1},//0.9
        };
 
        double[][] tA = new double[][] {
            { 0.2,  0.3,  0.3,  0.9,  0.7,  0.6,  0.9,  0.9,  1.0,  0.9,  //Addition
              0.0,  0.0,  0.1,  0.1,  0.1,  0.4,  0.3,  0.5,  0.3,  0.6,  //Subtraction
              0.0,  0.02,  0.02,  0.18,  0.12,  0.05,  0.18,  0.14,  1.16,  0.0,  //Multiplication
              0.0,  1.0,  0.4,  1.0,  0.4444,  0.6250,  0.8571,  0.5,  0.8333,  0.9 } //Division
        };
        
        SimpleMatrix trainingInputSet = new SimpleMatrix(tS);
        SimpleMatrix trainingAnswers = new SimpleMatrix(tA);
        
       // System.out.println("Training our simple neural network over 10 000 iterations");
        
        multiLayeredNN.train(trainingInputSet, trainingAnswers.transpose(), 600000);
        
        //System.out.print("Weights After training: ");
        //multiLayeredNN.PrintSynapseWeights();
        double totalError = 0.0;
        double ErrorAdd = 0.0, ErrorSub = 0.0, ErrorMul = 0.0, ErrorDiv = 0.0;
        int iterations = 10000;
        for(int i = 0; i < iterations; i++) { // Division
            
            double z, x, y;
            do {
            x = random.nextDouble();
            y = random.nextDouble();
            
            z = x / y;
            
            }while(z > 1 || z < 0);
            
        double[][] t = new double[][] {{x,y,0, 0, 0 , 1}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        double output = multiLayeredNN.think(thinkSet).getLast().get(0);
        ErrorDiv += Math.abs(output - z);
        
        //System.out.println(x + " - " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        ErrorDiv = ErrorDiv / iterations;
        
        for(int i = 0; i < iterations; i++) { // multiplication
            
            double z, x, y;
            do {
            x = random.nextDouble();
            y = random.nextDouble();
            
            z = x * y;
            
            }while(z > 1 || z < 0);
            
        double[][] t = new double[][] {{x,y,0, 0, 1, 0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        double output = multiLayeredNN.think(thinkSet).getLast().get(0);
        ErrorMul += Math.abs(output - z);
        
        //System.out.println(x + " - " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        ErrorMul = ErrorMul / iterations;
        
        for(int i = 0; i < iterations; i++) { // addition
            
            double z, x, y;
            do {
            x = random.nextDouble();
            y = random.nextDouble();
            
            z = x + y;
            
            }while(z > 1 || z < 0);
            
        double[][] t = new double[][] {{x,y,1, 0, 0, 0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        double output = multiLayeredNN.think(thinkSet).getLast().get(0);
        ErrorAdd += Math.abs(output - z);
        
        //System.out.println(x + " - " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        
        ErrorAdd = ErrorAdd / iterations;
        
        
        for(int i = 0; i < iterations; i++) { // subtraction
            
            double z, x, y;
            do {
            x = random.nextDouble();
            y = random.nextDouble();
            
            z = x - y;
            
            }while(z > 1 || z < 0);
            
        double[][] t = new double[][] {{x,y,0, 1, 0, 0}};
        SimpleMatrix thinkSet = new SimpleMatrix(t);
        double output = multiLayeredNN.think(thinkSet).getLast().get(0);
        ErrorSub += Math.abs(output - z);
        
        //System.out.println(x + " - " + y + " = "+ new DecimalFormat("#0.000000").format(output) + " Answer = " + z);
        
        }
        ErrorSub = ErrorSub / iterations;
        
        totalError = (ErrorAdd + ErrorDiv + ErrorMul + ErrorSub) / 4.0;
        
        System.out.println("Multi Layered Addition avg error: " + new DecimalFormat("#0.000000").format(ErrorAdd));
        System.out.println("Multi Layered Division avg error: " + new DecimalFormat("#0.000000").format(ErrorDiv));
        System.out.println("Multi Layered Multiplication avg error: " + new DecimalFormat("#0.000000").format(ErrorMul));
        System.out.println("Multi Layered Subtraction avg error: " + new DecimalFormat("#0.000000").format(ErrorSub));
        
        System.out.println("Multi Layered avg error: " + new DecimalFormat("#0.000000").format(totalError));
        
}

}
