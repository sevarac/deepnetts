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

import deepnetts.net.loss.LossType;
import deepnetts.net.train.Optimizers;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;

/**
 * This class represents output layer with sigmoid output function by default.
 * 
 * @author zoran
 */
public class OutputLayer extends AbstractLayer {

    protected float[] outputErrors;
    protected final String[] labels;      
    protected LossType lossType;    
    
    /**
     * Index position of target class on network output
     */
    int targetClassIdx;

    public OutputLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;        

        labels = new String[depth];
        // generate enumerated class names from 1..n
        for (int i = 0; i < depth; i++) {
            labels[i] = "Out_" + i;
        }
        
        setActivationType(ActivationType.SIGMOID);
    }
    
    public OutputLayer(int width, ActivationType activationFunction) {
        this.width = width;
        this.height = 1;
        this.depth = 1;        

        labels = new String[depth];
        // generate enumerated class names from 1..n
        for (int i = 0; i < depth; i++) {
            labels[i] = "Out_" + i;
        }
        
        setActivationType(activationFunction);
    }    

    public OutputLayer(String[] labels) {
        this.width = labels.length;
        this.height = 1;
        this.depth = 1;        
        this.labels = labels;
        setActivationType(ActivationType.SIGMOID);
    }
    
    public OutputLayer(String[] labels, ActivationType activationFunction) {
        this(labels);
        setActivationType(activationFunction);
    }    
    
    public final void setOutputErrors(final float[] outputErrors) {
        this.outputErrors = outputErrors;
    }    

    public final float[] getOutputErrors() {
        return outputErrors;
    }    

    public final LossType getLossType() {
        return lossType;
    }

    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }    
    
    @Override
    public void init() {
        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        outputErrors = new float[width];
        deltas = new Tensor(width);

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
     *
     * Calculates weighted input and layer outputs using sigmoid function.
     */
    @Override
    public void forward() {                
        outputs.copyFrom(biases);  // reset output to bias value        
        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {  // for all neurons in this layer  | ForkJoin split this in two until you reach size which makes sense: number of calculations = inputCols * outputCols           
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // add weighted sum
            }                        
            //outputs.set(outCol, ActivationFunctions.sigmoid(outputs.get(outCol)));      // apply activation function - could be tanh too
            outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol)));
        }             
    }

    /**
     * This method implements backward pass for the output layer. 
     * TODO: set externaly whcih loss function i used and correct  line 82 - cold be done in super class
     * If CE loss is used output error should not be multiplied with prime
     */
    @Override
    public void backward() {
        if (!batchMode) {   // reset delta weights and deltaBiases to zero in ezch iteration if not in batch/minibatch mode       
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }
        
        for (int dCol = 0; dCol < deltas.getCols(); dCol++) { // iterate all output neurons / deltas
            // TODO: da li ovde ttreba mnoziti sa izvodom? za CE ne treba, raspisi postupno kompletne matematicke formule za MSE i CE, Binary i MultiClass
            // ako je funkcija greske binary CE onda ne treba, a ako je MSE ond treba
            // kako ovde da znam da koja je funkcija greske - prilikom kreiranja mreze (build) da se setuje neki parametar
            if (lossType == LossType.MEAN_SQUARED_ERROR) {
                deltas.set(dCol, outputErrors[dCol] * ActivationFunctions.prime(activationType, outputs.get(dCol))); // delta = e*f1 
            } else {
                deltas.set(dCol, outputErrors[dCol]); // zato jer se u matematickom obliku skracuju - daj referencu!
            }
                        
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) { // prev layer is allways FullyConnected
                final float grad = deltas.get(dCol) * inputs.get(inCol);
                final float deltaWeight = Optimizers.sgd(learningRate, grad);
                //final float deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, dCol));                
                deltaWeights.add(inCol, dCol, deltaWeight ); // sum deltaWeight for batch mode
            }
            //deltaBiases[dCol] += -learningRate * deltas.get(dCol) + momentum * prevDeltaBiases[dCol];
              deltaBiases[dCol] += Optimizers.sgd(learningRate, deltas.get(dCol));
//            deltaBiases[dCol] += Optimizers.momentum(learningRate, deltas.get(dCol), momentum, prevDeltaBiases[dCol]);
        }
    }

    /**
     * Applies weight changes after one learning iteration or batch
     */
    @Override
    public void applyWeightChanges() {
        if (batchMode) { // if batch mode calculate average delta weights using batch samples (mini batch)
            deltaWeights.div(batchSize);
            Tensor.div(deltaBiases, batchSize);
        }

        // save current as prev delta weights (required for momentum)
        Tensor.copy(deltaWeights, prevDeltaWeights); 
        // apply(add) delta weights
        weights.add(deltaWeights);

        // save current as prev delta biases
        Tensor.copy(deltaBiases, prevDeltaBiases); 
        // apply(add) delta bias
        Tensor.add(biases, deltaBiases);

        if (batchMode) {    // for batch mode set all delta weights and biases to zero after applying changes. For online mode they are reseted in backward pass
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }
    }

}