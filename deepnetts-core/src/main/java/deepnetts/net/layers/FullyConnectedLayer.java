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
 * Fully connected layer has a single row of neurons connected to all neurons in
 * previous and next layer.
 *
 * Next layer can be fully connected or output 
 * Previous layer can be fully
 * connected, input, convolutional or max pooling
 *
 * @author Zoran Sevarac
 */
public class FullyConnectedLayer extends AbstractLayer {

    private static Logger LOG = Logger.getLogger(DeepNetts.class.getName());
    
    /**
     * Creates an instance of fully connected layer with specified width (number of neurons) and sigmoid activation function. 
     * 
     * @param width layer width / number of neurons in this layer
     */
    public FullyConnectedLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;
        this.activationType = ActivationType.SIGMOID;
    }
    
    /**
     * Creates an instance of fully connected layer with specified width (number of neurons) and activation function. 
     * 
     * @param width layer width / number of neurons in this layer
     * @param activationFunction activation function to use with this layer
     * @see ActivationFunctions
     */    
    public FullyConnectedLayer(int width, ActivationType activationFunction) {
        this(width);
        this.activationType = activationFunction;
    }    
    
     
    /**
     * Creates all data strucutres:  inputs, weights, biases, outputs, deltas, deltaWeights, deltaBiases
     * prevDeltaWeights, prevDeltaBiases. Init weights and biases
     */
    @Override
    public void init() {
        // check prev and next layers and throw exception if its illegall architecture
        // if next layer is conv or max throw exception UnsupportedArchitectureException - nema smisla

        inputs = prevLayer.outputs;        
        outputs = new Tensor(width);
        deltas = new Tensor(width);
        
         if (prevLayer instanceof FullyConnectedLayer) { // ovo ako je prethodni 1d layer, odnosno ako je prethodni fully connected        
            weights = new Tensor(prevLayer.width, width);
            deltaWeights = new Tensor(prevLayer.width, width);
            prevDeltaWeights = new Tensor(prevLayer.width, width);
            prevGradSums = new Tensor(prevLayer.width, width);  // for AdaGrad
            prevBiasSums = new Tensor(width);
            
            WeightsInit.xavier(weights.getValues(), prevLayer.width, width);
           // WeightsInit.randomize(weights.getValues());
            
        } else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer)|| (prevLayer instanceof InputLayer)) { // ako je pooling ili konvolucioni 2d ili 3d       
            weights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width); // ovde bi trebalo: prevLayer.width, prevLayer.height, width, prevLayer.depth ili prevLayer.width, prevLayer.height, prevLayer.depth, width,     
            deltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            prevDeltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width); 
            prevGradSums = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            prevBiasSums = new Tensor(width);  

            int totalInputs = prevLayer.getWidth() * prevLayer.getHeight() * prevLayer.getDepth();
            WeightsInit.xavier(weights.getValues(), totalInputs, width);
        }

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];
        WeightsInit.randomize(biases);
    }

    @Override
    public void forward() {
        // if previous layer is FullyConnected
        if (prevLayer instanceof FullyConnectedLayer) { 
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {           // for all neurons/outputs in this layer
                // calculate weighted sums of inputs
                outputs.set(outCol, biases[outCol]);                               // first use (add) bias
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {            // iterate all inputs from prev layer
                    outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));  // add weighted sum to outputs
                } 

                // apply activation function to all weigthed sums stored in outputs
                outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol))); // ovo bi mogla da bude lambda ili Function()
                // parallel apply activation function on each eleemnt of array
            }
        }
        
        // if previous layer is MaxPooling, Convolutional or input layer (2D or 3D) - TODO: posto je povezanost svi sa svima ovo mozda moze i kao 1d na 1d niz, verovatno je efikasnije
        else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) { // povezi sve na sve                      
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {          // for all neurons/outputs in this layer
                // calculate weighted sums of inputs
                outputs.set(outCol, biases[outCol]);                              // first add (set to) bias
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {   // iterate depth from prev/input layer
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {      // iterate current channel by height (rows)
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) {   // iterate current feature map by width (cols)
                            outputs.add(outCol, inputs.get(inRow, inCol, inDepth) * weights.get(inCol, inRow, inDepth, outCol)); // add to weighted sum of all inputs (TODO: ako je svaki sa svima to bi mozda moglo da bude i jednostavno i da se prodje u jednom loopu a ugnjezdeni loopovi bi se lakse paralelizovali)
                        }
                    }
                }                
                // apply activation function to all weigthed sums stored in outputs 
                outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol)));
            }
        }                
    }
    
    @Override
    public void backward() {
        if (!batchMode) { // if online mode reset deltaWeights and deltaBiases to zeros
            deltaWeights.fill(0); 
            Arrays.fill(deltaBiases, 0);
        }
        
        deltas.fill(0); // reset current delta     

        // STEP 1. propagate weighted deltas from next layer (which can be output or fully connected) and calculate deltas for this layer
        for (int dCol = 0; dCol < deltas.getCols(); dCol++) {   // for every neuron/delta in this layer           
            for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { // iterate all deltas from next layer
                deltas.add(dCol, nextLayer.deltas.get(ndCol) * nextLayer.weights.get(dCol, ndCol)); // calculate weighted sum of deltas from the next layer
            }
                        
            final float delta = deltas.get(dCol) * ActivationFunctions.prime(activationType, outputs.get(dCol));
            deltas.set(dCol, delta); 
        } // end sum weighted deltas from next layer
            
        
        // STEP 2. calculate delta weights if previous layer is FC (2D weights matrix) 
        if ((prevLayer instanceof FullyConnectedLayer)) {
            for (int dCol = 0; dCol < deltas.getCols(); dCol++) { // this iterates neurons (weights depth)
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                    final float grad = deltas.get(dCol) * inputs.get(inCol); // gradient dE/dw
                    
                    float deltaWeight=0;
                    switch(optimizer) {
                        case SGD:
                                deltaWeight = Optimizers.sgd(learningRate, grad);
                            break;
                        case MOMENTUM:
                            //deltaWeight = Optimizers.sgd(learningRate, grad);
                             deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, dCol));
                            break;
                        case ADAGRAD :
                                prevGradSums.add(inCol, dCol, grad*grad);
                                deltaWeight = Optimizers.adaGrad(learningRate, grad, prevGradSums.get(inCol, dCol));
                            break;
                    }
                                            
                    deltaWeights.add(inCol, dCol, deltaWeight);
                }
                
               float deltaBias=0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(dCol));
                        break;
                    case MOMENTUM:
                        //deltaBias = Optimizers.sgd(learningRate, deltas.get(dCol));
                        deltaBias = Optimizers.momentum(learningRate, deltas.get(dCol), momentum, prevDeltaBiases[dCol]);
                        break;
                    case ADAGRAD:
                        prevBiasSums.add(dCol, deltas.get(dCol)*deltas.get(dCol));
                        deltaBias = Optimizers.adaGrad(learningRate, deltas.get(dCol), prevBiasSums.get(dCol));                        
                        break;
                }
                
                deltaBiases[dCol] += deltaBias;
            }
        }
        
        else if ((prevLayer instanceof InputLayer) || 
                 (prevLayer instanceof ConvolutionalLayer) || 
                 (prevLayer instanceof MaxPoolingLayer)) {
            
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // for all neurons/deltas in this layer
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) { // iterate all inputs from previous layer
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) { 
                            float grad = deltas.get(deltaCol) * inputs.get(inRow, inCol, inDepth);  // da li je ovde greska treba ih sumitrati sve tri po dubini  // da li ove ulaze treba sabirati??? jer jedna celike ima ulaze iz tri prethodn akanala?
                            //float deltaWeight = -learningRate * grad + momentum * prevDeltaWeights.get(inCol, deltaCol,  inRow, inDepth); // with momentum
                            //float deltaWeight = Optimizers.sgd(learningRate, grad);           
                           // float deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, deltaCol,  inRow, inDepth));  
                           // (Optimizers::sgd, learningRate, grad) (methodRef, float ... )
                            float deltaWeight=0;
                            switch(optimizer) {
                                case SGD:
                                        deltaWeight = Optimizers.sgd(learningRate, grad);
                                    break;
                                case MOMENTUM:
                                     //deltaWeight = Optimizers.sgd(learningRate, grad);
                                       deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, inRow, inDepth, deltaCol));// 
                                    break;
                                case ADAGRAD:
                                        prevGradSums.add(inCol, deltaCol,  inRow, inDepth, grad*grad);
                                        deltaWeight = Optimizers.adaGrad(learningRate, grad, prevGradSums.get(inCol, inRow, inDepth, deltaCol));                                                                         
                                    break;
                            }                           
                            
                            deltaWeights.add(inCol, inRow, inDepth, deltaCol, deltaWeight);                      
                        }
                    }
                }            
                
              float deltaBias=0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                        break;
                    case MOMENTUM:
                     //   deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                          deltaBias = Optimizers.momentum(learningRate, deltas.get(deltaCol), momentum, prevDeltaBiases[deltaCol]);
                        break;
                    case ADAGRAD:
                        prevBiasSums.add(deltaCol, deltas.get(deltaCol)*deltas.get(deltaCol));
                        deltaBias = Optimizers.adaGrad(learningRate, deltas.get(deltaCol), prevBiasSums.get(deltaCol));                        
                        break;
                }
                
                //deltaBiases[deltaCol] += -learningRate * deltas.get(deltaCol) + momentum * prevDeltaBiases[deltaCol];                                         
                //deltaBiases[deltaCol] += Optimizers.sgd(learningRate, deltas.get(deltaCol));
                deltaBiases[deltaCol] += deltaBias;
            }
        }
    }

    @Override
    public void applyWeightChanges() {
        if (batchMode) { // podeli Delta weights sa brojem uzoraka odnosno backward passova
            deltaWeights.div(batchSize);
            Tensor.div(deltaBiases, batchSize);
        }        
        
        if (prevLayer instanceof FullyConnectedLayer) {            
            Tensor.copy(deltaWeights, prevDeltaWeights); // save as prev delta weight
            Tensor.copy(deltaBiases, prevDeltaBiases);
            
            weights.add(deltaWeights);        
            Tensor.add(biases, deltaBiases);                        
        }
                
        //  ovo je kad je povezan na 2D ili 3d sloj iza - onde je weights 3d ili 4d verovatno i ovo moze samo sa Tensor.add i Tensor.copy za prevDeltaWeights
        if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) {
//            for (int wFourth = 0; wFourth < weights.getFourthDim(); wFourth++) { 
//                for (int wDepth = 0; wDepth < weights.getDepth(); wDepth++) { 
//                    for (int wRow = 0; wRow < weights.getRows(); wRow++) {                    
//                         for (int wCol = 0; wCol < weights.getCols(); wCol++) {                          
//                            prevDeltaWeights.set(wRow, wCol, wDepth, wFourth, deltaWeights.get(wRow, wCol, wDepth, wFourth));    // remmember as prev change in order to use it for momentum                                                         
//                            weights.add(wRow, wCol, wDepth, wFourth, deltaWeights.get(wRow, wCol, wDepth, wFourth)); // can also use it as single for sub, but prev will get lost
//                        }
//                    }
//                }
//            }
//          ovo radi bolje nego ovo iznad a trebalo bi da je isto            
            Tensor.copy(deltaWeights, prevDeltaWeights); // ove dve se mogu raditi paraleln i nezavisno
            weights.add(deltaWeights);
      
            Tensor.copy(deltaBiases, prevDeltaBiases);
            Tensor.add(biases, deltaBiases);
            
        }
        
        if (batchMode) {
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }
        
        
    }

}