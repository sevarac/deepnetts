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
import deepnetts.net.MultiLayerPerceptron;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Iris Classification Problem.
 * This example is using Softmax activation in output addLayer and Cross Entropy Loss function.
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class IrisClassificationCE {
    
    public static void main(String[] args) throws DeepNettsException, IOException {
        
        // create instance of multi addLayer percetpron using builder
        MultiLayerPerceptron convNet = MultiLayerPerceptron.builder()
                                            .addInputLayer(4)
                                            .addFullyConnectedLayer(20, ActivationType.TANH)
                                            .addFullyConnectedLayer(13)
                                            .addOutputLayer(3, SoftmaxOutputLayer.class)
                                            .lossFunction(LossType.CROSS_ENTROPY)
                                            .randomSeed(123).
                                        build();

        // load irisdata  set
        DataSet dataSet = loadIrisDataSet();
              
        // create and configure instanceof backpropagation trainer 
        BackpropagationTrainer trainer = new BackpropagationTrainer(convNet);
        trainer.setMaxError(0.02f);
        trainer.setLearningRate(0.5f);
        trainer.setMomentum(0.3f);
        trainer.setBatchMode(false);
        //trainer.setBatchSize(150);
        trainer.setMaxIterations(10000);
        trainer.train(dataSet);                                                                                                                
    }
    
    
    public static DataSet loadIrisDataSet() throws FileNotFoundException, IOException {
        DataSet dataSet = new DataSet(); // DataSetInterface(4, 3)
        
        BufferedReader br = new BufferedReader(new FileReader(new File("datasets/iris_data_normalised.txt")));
        String line = "";
        while((line = br.readLine()) != null) {
           String[] values = line.split(",");
           float[] inputs =  new float[4];
           float[] outputs = new float[3];
           for(int i=0; i<4; i++) {
               inputs[i] = Float.parseFloat(values[i]);      
           }
           
           for(int j=0; j<3; j++) {
               outputs[j] = Float.parseFloat(values[4+j]);
           }
           
           dataSet.add(new BasicDataSetItem(inputs, outputs));                      
        }
                
        return dataSet;        
    }    
    
}
