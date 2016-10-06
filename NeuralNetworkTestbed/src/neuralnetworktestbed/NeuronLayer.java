package neuralnetworktestbed;
import java.util.Random;

import org.ejml.simple.*;
import org.ejml.data.*;
import org.ejml.factory.*;



public class NeuronLayer {
    
    SimpleMatrix m_SynapsWeights; 
    private Random m_Random;

    
    public NeuronLayer(int numberOfNeurons, int inputsPerNeuron) {
        
        m_Random = new Random(0);
        
        double[][] startWeights = new double[inputsPerNeuron][numberOfNeurons];
        
        for(int row = 0; row < inputsPerNeuron; row++) {
            for(int col = 0; col < numberOfNeurons; col++) {
                startWeights[row][col] = (m_Random.nextDouble() - 0.5) * 2.0;
            }
        }
        
        m_SynapsWeights = new SimpleMatrix(startWeights);
    }
}
