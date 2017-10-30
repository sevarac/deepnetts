/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */
    
package deepnetts.net.layers;

import deepnetts.core.DeepNetts;
import deepnetts.net.train.Optimizers;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;
import java.util.logging.Logger;

/**
 *
 * @author zoran
 */
public class SoftmaxOutputLayer extends OutputLayer {

    /**
     * Index position of target class on network output
     */
    int targetClassIdx;

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    public SoftmaxOutputLayer(int width) {
        super(width);
        setActivationType(ActivationType.SOFTMAX);
    }

    public SoftmaxOutputLayer(String[] labels) {
        super(labels);
        setActivationType(ActivationType.SOFTMAX);
    }

    // ovde ide inicijalizacija koja nije moguca pre nego se layer ubaci u mrezu i zna ko mi je prethodni
    @Override
    public void init() {
        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        outputErrors = new float[width];
        deltas = new Tensor(width);
//        derivatives = new Tensor(width);

        // height je koliko ima neurona u prethodnom FC lejeru  - pretpostavka je da moze samo FC lejer da bude iza
        int prevLayerWidth = prevLayer.getWidth();
        weights = new Tensor(prevLayerWidth, width);
        deltaWeights = new Tensor(prevLayerWidth, width);
        prevDeltaWeights = new Tensor(prevLayerWidth, width);
        WeightsInit.xavier(weights.getValues(), prevLayerWidth, width);

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];
        WeightsInit.randomize(biases);
    }

    /**
     * This method implements forward pass for the output layer. 
     * Calculates layer outputs using softmax function
     */
    @Override
    public void forward() {
        // find max weightedSum
        float maxWs = Float.NEGATIVE_INFINITY;

        //  compute weighted sums (activations) and find max weighted sum
        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {                    // for all neurons in this layer                
            outputs.set(outCol, biases[outCol]);                                        // first add bias
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {                    // iterate all inputs
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // add weighted sum of inputs
            }

            if (outputs.get(outCol) > maxWs) { // find max weighted sum
                maxWs = outputs.get(outCol);
            }
        }

        // calculate outputs and denominator sum (use max for numerical stability)
        float denSum = 0;
        for (int col = 0; col < outputs.getCols(); col++) {
            outputs.set(col, (float)Math.exp(outputs.get(col) - maxWs)); // maxWs used for numerical stability
            denSum += outputs.get(col);
        }

        // TODO: add div opeation to matrix instead loop below:     outputs.div(denSum);
        
        // scale all outputs to sum to 1
        for (int col = 0; col < outputs.getCols(); col++) {
            outputs.set(col, outputs.get(col) / denSum);          
        }
      
    }

    /**
     *
     */
    @Override
    public void backward() {
        if (!batchMode) {
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        float derivative = 0, deltaWeight = 0;

        for (int outCol = 0; outCol < outputs.getCols(); outCol++) { // iterate all output neurons / deltas

            if (outCol == targetClassIdx) {
                derivative = outputs.get(outCol) * (1 - outputs.get(outCol));
            } else {
                derivative = -outputs.get(targetClassIdx) * outputs.get(outCol);
            }

            // pitanje: zasto ovde nije outputErrors[outIdx] ? prouci teoriju! - ovo je kljucna caka za se: propagira se samo greska target neurona!!!
            deltas.set(outCol, outputErrors[targetClassIdx] * derivative); // e*f1  // da li svi neuroni treba da se mnoze sa  outputErrors.get(targetClassIdx) ??? to je verovatno problem...             

            for (int inCol = 0; inCol < inputs.getCols(); inCol++) { // prev layer is allways FullyConnected. iterate all inputs/weights for the current neuron         
                float grad = deltas.get(outCol) * inputs.get(inCol);
               // deltaWeight = -learningRate * grad + momentum * prevDeltaWeights.get(inCol, outCol);
                deltaWeight = Optimizers.sgd(learningRate, grad);
                //deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, outCol));
                deltaWeights.add(inCol, outCol, deltaWeight); // 
            }

            // deltaBiases[outCol] += -learningRate * deltas.get(outCol) + momentum * prevDeltaBiases[outCol];
            deltaBiases[outCol] += Optimizers.sgd(learningRate, deltas.get(outCol));
            //deltaBiases[outCol] += Optimizers.momentum(learningRate, deltas.get(outCol), momentum, prevDeltaBiases[outCol]);
        }
    }
        
    // ova metod amoze dda ide u OutputLayer class
    @Override
    public void applyWeightChanges() {

        if (batchMode) { // podeli Delta weights sa brojem uzoraka odnosno backward passova ako je u batch modu
            deltaWeights.div(batchSize);
            Tensor.div(deltaBiases, batchSize);
        }

        Tensor.copy(deltaWeights, prevDeltaWeights); // save as prev delta weight
        weights.add(deltaWeights);

        Tensor.copy(deltaBiases, prevDeltaBiases);
        Tensor.add(biases, deltaBiases);

        if (batchMode) {
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }
    }
       

    public void setTargetClassIdx(int targetClassIdx) {
        this.targetClassIdx = targetClassIdx;
    }
    
}
