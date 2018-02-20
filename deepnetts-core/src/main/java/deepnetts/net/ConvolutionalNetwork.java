/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.net;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Convolutional neural network
 *
 * @author Zoran Sevarac
 */
public class ConvolutionalNetwork extends NeuralNetwork implements Serializable {

    private ConvolutionalNetwork() {
        super();
    }

    public static ConvolutionalNetwork.Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private final ConvolutionalNetwork neuralNet = new ConvolutionalNetwork();

        private ActivationType defaultActivationType = ActivationType.RELU;
        private Class<CrossEntropyLoss> defaultLossFunction = CrossEntropyLoss.class;

        /**
         * Input layer with specified width and height, and 3 channels by 
         * default.
         *
         * @param width
         * @param height
         * @return
         */
        public Builder addInputLayer(int width, int height) {
            InputLayer inLayer = new InputLayer(width, height, 3);
            neuralNet.setInputLayer(inLayer);
            neuralNet.addLayer(inLayer);

            return this;
        }

        /**
         * Input layer with specified width, height and number of channels.
         * 
         * @param width
         * @param height
         * @param channels
         * @return 
         */
        public Builder addInputLayer(int width, int height, int channels) {
            InputLayer inLayer = new InputLayer(width, height, channels);
            neuralNet.setInputLayer(inLayer);
            neuralNet.addLayer(inLayer);

            return this;
        }

        public Builder addFullyConnectedLayer(int width) {
            FullyConnectedLayer layer = new FullyConnectedLayer(width);
            neuralNet.addLayer(layer);
            return this;
        }

        public Builder addFullyConnectedLayer(int width, ActivationType activationFunction) {
            FullyConnectedLayer layer = new FullyConnectedLayer(width, activationFunction);
            neuralNet.addLayer(layer);
            return this;
        }

        /**
         * Adds output layer with specified width and soft max activation
         * function
         *
         * @param width layer width
         * @return builder instance
         */
        public Builder addOutputLayer(int width) { // ActivationType.SOFTMAX
            OutputLayer outputLayer = null;
            if (width == 1) {
                outputLayer = new OutputLayer(width);
            } else {
                outputLayer = new SoftmaxOutputLayer(width);
            }
            neuralNet.setOutputLayer(outputLayer);
            neuralNet.addLayer(outputLayer);

            return this;
        }

        public Builder addOutputLayer(int width, Class<? extends OutputLayer> clazz) { // ActivationType.SOFTMAX
            try {
                OutputLayer outputLayer = clazz.getDeclaredConstructor(Integer.TYPE).newInstance(width);
                neuralNet.addLayer(outputLayer);
                neuralNet.setOutputLayer(outputLayer);
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

            neuralNet.setOutputLayer(outputLayer);
            neuralNet.addLayer(outputLayer);

            return this;
        }

        // stride???
        public Builder addConvolutionalLayer(int filterSize, int channels) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(filterSize, filterSize, channels);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        public Builder addConvolutionalLayer(int filterSize, int channels, ActivationType activationType) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(filterSize, filterSize, channels, activationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        public Builder addConvolutionalLayer(int filterWidth, int filterHeight, int channels) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(filterWidth, filterHeight, channels);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        public Builder addConvolutionalLayer(int filterWidth, int filterHeight, int channels, ActivationType activationType) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(filterWidth, filterHeight, channels, activationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        public Builder addMaxPoolingLayer(int filterSize, int stride) {
            MaxPoolingLayer poolingLayer = new MaxPoolingLayer(filterSize, filterSize, stride);
            neuralNet.addLayer(poolingLayer);
            return this;
        }

        public Builder addMaxPoolingLayer(int filterWidth, int filterHeight, int stride) {
            MaxPoolingLayer poolingLayer = new MaxPoolingLayer(filterWidth, filterHeight, stride);
            neuralNet.addLayer(poolingLayer);
            return this;
        }

        public Builder activationFunction(ActivationType activationType) {
            this.defaultActivationType = activationType;
            return this;
        }

        public Builder withLossFunction(Class<? extends LossFunction> clazz) {
            try {
                LossFunction loss = clazz.getDeclaredConstructor(NeuralNetwork.class).newInstance(neuralNet);
                neuralNet.setLossFunction(loss);
            } catch (NoSuchMethodException | SecurityException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                Logger.getLogger(ConvolutionalNetwork.class.getName()).log(Level.SEVERE, null, ex);
            }

            return this;
        }

        public Builder withLossFunction(LossType lossType) {
            LossFunction loss = null;
            switch (lossType) {
                case MEAN_SQUARED_ERROR:
                    loss = new MeanSquaredErrorLoss(neuralNet);
                    break;
                case CROSS_ENTROPY:
                    if (neuralNet.getOutputLayer().getWidth() == 1) {
                        loss = new BinaryCrossEntropyLoss(neuralNet);
                    } else {
                        loss = new CrossEntropyLoss(neuralNet);
                    }
                    break;
            }
            neuralNet.setLossFunction(loss);

            return this;
        }

        public Builder withRandomSeed(long seed) {
            RandomGenerator.getDefault().initSeed(seed);
            return this;
        }

        public ConvolutionalNetwork build() {
            // connect and init layers, weights matrices etc.
            AbstractLayer prevLayer = null;

            for (int i = 0; i < neuralNet.getLayers().size(); i++) {
                AbstractLayer layer = neuralNet.getLayers().get(i);
                if (!(layer instanceof OutputLayer)) { // ne za izlazni layer
                    layer.setActivationType(defaultActivationType);
                }
                layer.setPrevLayer(prevLayer);
                if (prevLayer != null) {
                    prevLayer.setNextlayer(layer);
                }
                prevLayer = layer; // current layer becomes prev layer in next iteration                            
            }

            // init all layers
            neuralNet.getLayers().forEach((layer) -> {
                layer.init();
            });

            // if loss is not set use default loss function
            if (neuralNet.getLossFunction() == null) {
                withLossFunction(defaultLossFunction);
            }

            return neuralNet;
        }
    }

    /**
     * Returns all weights from this network as list of strings.
     * TODO; for convolutional layer get filter weights
     * 
     * @return
     */
    public List<String> getWeights() {
        List weightsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
            if (layer instanceof ConvolutionalLayer) {
                Tensor[] filters = ((ConvolutionalLayer)layer).getFilters();
                String filterStr = Tensor.valuesAsString(filters);
                weightsList.add(filterStr);
            } else {
                weightsList.add(layer.getDeltaWeight().toString());
            }
        } 
        return weightsList;
    }
    
    public void setWeights(List<String> weights) {
        int weightsIdx=0;
        
        for (int layerIdx = 1; layerIdx < getLayers().size(); layerIdx++) {
            AbstractLayer layer = getLayers().get(layerIdx);
            if (layer instanceof ConvolutionalLayer) {
                ((ConvolutionalLayer)layer).setFilters(weights.get(weightsIdx));
                weightsIdx++;
            } else if (layer instanceof FullyConnectedLayer || layer instanceof OutputLayer) {
                layer.setWeights(weights.get(weightsIdx));
                weightsIdx++;
            } else {
                continue;
            }
        }
    }


    

    public List<String> getDeltaWeights() {
        List weightsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
                weightsList.add(layer.getDeltaWeight().toString());
        }
        return weightsList;
    }

    // returns outputs for all layers
    public List<String> getAllOutputs() {
        List outputsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
            outputsList.add(layer.getOutputs());
        }
        return outputsList;
    }

}
