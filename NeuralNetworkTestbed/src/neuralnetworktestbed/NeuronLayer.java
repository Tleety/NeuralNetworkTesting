package neuralnetworktestbed;
import java.util.Random;

import org.ejml.simple.*;



public class NeuronLayer {
    
    SimpleMatrix m_SynapsWeights; 
    private Random m_Random;

    public int m_Rows, m_Columns;
    
    public NeuronLayer(int inputsPerNeuron, int numberOfNeurons) {
        
        m_Rows = inputsPerNeuron;
        m_Columns = numberOfNeurons;
        
        m_Random = new Random();
        
        double[][] startWeights = new double[inputsPerNeuron][numberOfNeurons];
        
        for(int row = 0; row < inputsPerNeuron; row++) {
            for(int col = 0; col < numberOfNeurons; col++) {
                startWeights[row][col] = (m_Random.nextDouble() - 0.5) * 2.0;
            }
        }
        
        m_SynapsWeights = new SimpleMatrix(startWeights);
    }
    
    
    public void Rebalance(int inputsPerNeuron, int numberOfNeurons) {
        
       // m_SynapsWeights.print();
        
        double[][] newWeights = new double[inputsPerNeuron][numberOfNeurons];

        for(int row = 0; row < inputsPerNeuron; row++) {
            for(int col = 0; col < numberOfNeurons; col++) {

                if(row >= m_Rows || col >= m_Columns) {
                    newWeights[row][col] = (m_Random.nextDouble() - 0.5) * 2.0;
                } else {
                    newWeights[row][col] = m_SynapsWeights.get(row, col);
                }
            }
        }

        m_SynapsWeights = new SimpleMatrix(newWeights);

        m_Rows = inputsPerNeuron;
        m_Columns = numberOfNeurons;
        
       // m_SynapsWeights.print();
    }
}
