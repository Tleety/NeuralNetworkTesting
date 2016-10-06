
package neuralnetworktestbed;

import java.util.Random;
import org.ejml.simple.*;

public class NeuralLayer {
    SimpleMatrix synapsWeights;
    public NeuralLayer(int nrOfNeurons, int nrOfInputsPerNeuron) {
        synapsWeights = new SimpleMatrix(nrOfInputsPerNeuron, nrOfNeurons);
        for(int row = 0; row < nrOfInputsPerNeuron; row++) {
            for(int col = 0; col < nrOfNeurons; col++) {
                Random rand = new Random();
                double r = 2 * rand.nextDouble() - 1;
                synapsWeights.set(row, col, r);
            }
        }
    }
}
