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

package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.net.MultiLayerPerceptron;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;

/**
 * Minimal example for linear regression using MultiLayerPerceptron.
 * Fits a straight line through the data.
 Uses a single addLayer with one output and linear activation function, and Mean Squared Error for Loss function.
 You can use linear regression to roughly estimate a global trend in data.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LinearRegression {
    
    public static void main(String[] args) {
        
        DataSet dataSet =null; // get data from some file or method        
        
        NeuralNetwork neuralNet = MultiLayerPerceptron.builder()
                                    .addInputLayer(5)
                                    .addOutputLayer(1, ActivationType.LINEAR)
                                    .lossFunction(LossType.MEAN_SQUARED_ERROR)
                                    .build();
                       
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
                               trainer.setLearningRate(0.1f)
                                      .train(dataSet);           
    }
    
}
