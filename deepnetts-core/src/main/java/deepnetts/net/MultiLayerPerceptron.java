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
    
package deepnetts.net;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.ActivationFunctions;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.util.WeightsInit;
import java.lang.reflect.InvocationTargetException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Multi Layer Perceptron neural network architecture.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MultiLayerPerceptron extends NeuralNetwork {
    
    /**
     * Private constructor allows instantiation only using builder
     */
    private MultiLayerPerceptron() {
        super();
    }
    
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        /**
         * MultiLayerPerceptron network that will be created and configured using this builder
         */
        private MultiLayerPerceptron network = new MultiLayerPerceptron();
       
        
        /**
         * Adds input addLayer with specified width to the network.
         * 
         * @param width addLayer width
         * @return builder instance
         */
        public Builder addInputLayer(int width) {
            InputLayer inLayer = new InputLayer(width, 1, 1);
            network.setInputLayer(inLayer);
            network.addLayer(inLayer);
            
            return this;
        }          
        
             
        /**
         * Adss fully connected addLayer with specified width and Sigmoid function to the network.
         * 
         * @param width addLayer width / number of neurons
         * @return builder instance
         */
        public Builder addFullyConnectedLayer(int width) {
            FullyConnectedLayer layer = new FullyConnectedLayer(width);
            network.addLayer(layer);
            return this;
        }
                
       /**
         * Adds fully connected addLayer with specified width and activation function to the network.
         * 
         * @param width addLayer width / number of neurons
         * @param activationFunction activation function to use for this addLayer         * 
         * @return builder instance
         * @see ActivationFunctions
         */        
        public Builder addFullyConnectedLayer(int width, ActivationType activationFunction) {
            FullyConnectedLayer layer = new FullyConnectedLayer(width, activationFunction);
            network.addLayer(layer);
            return this;
        }       
        
        public Builder addLayer(AbstractLayer layer) {
            network.addLayer(layer);
            return this;
        }                
            
        /**
         * Adds SoftMaxOutput Layer as output addLayer to the network
         * 
         * @param width  addLayer width / number of neurons
         * @return builder instance
         */
        public Builder addOutputLayer(int width) {
            SoftmaxOutputLayer outputLayer = new SoftmaxOutputLayer(width);
            network.setOutputLayer(outputLayer);
            network.addLayer(outputLayer);
            
            return this;
        }
        
        /**
         * Adds output addLayer of specified class to the network
 Output addLayer class can be SoftmaxOutputLayer or SigmoidOutputLayer
         * 
         * @param width  addLayer width / number of neurons
         * @param clazz output addLayer class 
         * @return builder instance
         */        
        public Builder addOutputLayer(int width, Class<? extends OutputLayer> clazz) {
            try {
                OutputLayer outputLayer = clazz.getDeclaredConstructor( Integer.TYPE).newInstance(width);
                network.setOutputLayer(outputLayer);
                network.addLayer(outputLayer);            
            } catch (InstantiationException | IllegalAccessException | NoSuchMethodException | SecurityException | IllegalArgumentException | InvocationTargetException ex) {
                Logger.getLogger(ConvolutionalNetwork.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            return this;
        }     

        public Builder addOutputLayer(int width, ActivationType activationType) {
            OutputLayer outputLayer = null;
            if (activationType.equals(ActivationType.SOFTMAX)) {
                outputLayer = new SoftmaxOutputLayer(width);
            } else {            
                outputLayer = new OutputLayer(width);
                outputLayer.setActivationType(activationType);
            }
            
            network.setOutputLayer(outputLayer);
            network.addLayer(outputLayer);
            
            return this;
        }

        
        /**
         * Adds specified loss function to the network.
         * Loss Function can be MSE or CE
         * 
         * @param clazz
         * @return 
         */
        public Builder lossFunction(Class<? extends LossFunction> clazz) {  
            try {            
                LossFunction loss = clazz.getDeclaredConstructor(NeuralNetwork.class).newInstance(network);
                network.setLossFunction(loss);
            } catch (NoSuchMethodException | SecurityException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                Logger.getLogger(ConvolutionalNetwork.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            return this;
        }
        
        public Builder lossFunction(LossType lossType) {  
            LossFunction loss = null;
            switch(lossType) {
                case MEAN_SQUARED_ERROR :
                        loss = new MeanSquaredErrorLoss(network);
                    break;
                case CROSS_ENTROPY:
                        if (network.getOutputLayer().getWidth() == 1)
                            loss = new BinaryCrossEntropyLoss(network);
                        else 
                            loss = new CrossEntropyLoss(network);
                    break;                        
            }
            network.setLossFunction(loss);
            return this;
        }        
        
        /**
         * Initializes random number generator with specified seed in order to get same random number sequences (used for weights initialization).
         * 
         * @param seed
         * @return 
         */
        public Builder randomSeed(long seed) {
            WeightsInit.initSeed(seed);
            return this;
        }
        
        
        public MultiLayerPerceptron build() {

            // prodji kroz celu mrezu i inicijalizuj matrice tezina / konekcije
            // povezi sve lejere
            AbstractLayer prevLayer = null;

            // povezi lejere i pozovi metodu init koja vrsi internu inicijalizaciju nakon sto je lejer povezan u mrezi
            for (int i = 0; i < network.getLayers().size(); i++) {
                AbstractLayer layer = network.getLayers().get(i);
                layer.setPrevLayer(prevLayer);
                if (prevLayer!= null) prevLayer.setNextlayer(layer);               
                prevLayer = layer;                 // current addLayer becomes prev addLayer for layerin next iteration                            
            }
                       
            for(AbstractLayer layer : network.getLayers()) {
               layer.init();
            }                    
            
            // throw excption if loss is null 
            
            
            return network;
        }        
        
    }
            

}
