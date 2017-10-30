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

import deepnetts.data.BasicDataSetItem;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.MultiLayerPerceptron;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;

/**
 * Solve XOR problem to confirm that backpropagation is working, and that it can solve the simplest nonlinear problem.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class XorExample {
    
    public static void main(String[] args) throws DeepNettsException {
        
        DataSet dataSet = xorDataSet();
        
        MultiLayerPerceptron convNet = MultiLayerPerceptron.builder()
                .inputLayer(2)
                .fullyConnectedLayer(3, ActivationType.TANH)
                .outputLayer(1, OutputLayer.class)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .randomSeed(123)
                .build();
                                           
        BackpropagationTrainer trainer = new BackpropagationTrainer(convNet);
        trainer.setMaxError(0.01f);
        trainer.setLearningRate(0.9f);
        trainer.train(dataSet);                                                                                                                
    }
    
    public static DataSet xorDataSet() {
        DataSet dataSet = new DataSet();
        
        DataSetItem item1 = new BasicDataSetItem(new float[] {0, 0}, new float[] {0});
        dataSet.add(item1);
        
        DataSetItem item2 = new BasicDataSetItem(new float[] {0, 1}, new float[] {1});
        dataSet.add(item2);
        
        DataSetItem item3 = new BasicDataSetItem(new float[] {1, 0}, new float[] {1});
        dataSet.add(item3);
        
        DataSetItem item4 = new BasicDataSetItem(new float[] {1, 1}, new float[] {0});
        dataSet.add(item4);     
        
        return dataSet;        
    }
    
}
