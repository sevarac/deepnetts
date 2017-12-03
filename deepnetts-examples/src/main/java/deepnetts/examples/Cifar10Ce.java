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

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.OptimizerType;
import deepnetts.util.DeepNettsException;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.loss.LossType;
import java.io.File;
import java.io.IOException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Cifar10Ce {
            
    int imageWidth = 32;
    int imageHeight = 32;
         
    String labelsFile = "datasets/cifar10/labels.txt";
    String trainingFile = "datasets/cifar10/train.txt";
   // String testFile = "datasets/cifar10/test.txt";         
    
    static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());        
        
    public void run() throws DeepNettsException, IOException {
        LOGGER.info("Loading images...");
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);        
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile), true, 100);
        imageSet.invert();
        imageSet.zeroMean();
        imageSet.shuffle();
            
        int labelsCount = imageSet.getLabelsCount();
        
        LOGGER.info("Done!");             
        LOGGER.info("Creating neural network...");

        ConvolutionalNetwork neuralNet = ConvolutionalNetwork.builder()
                                        .addInputLayer(imageWidth, imageHeight) 
                                        .addConvolutionalLayer(5, 5, 6, ActivationType.TANH)
                                        .addMaxPoolingLayer(2, 2, 2)                                 
                                        .addFullyConnectedLayer(30, ActivationType.TANH)     
                                        .addOutputLayer(labelsCount, ActivationType.SOFTMAX)
                                        .lossFunction(LossType.CROSS_ENTROPY)                
                                        .build();
        
        LOGGER.info("Done!");       
        LOGGER.info("Training neural network"); 
         
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setLearningRate(0.01f);
        trainer.setMaxError(0.5f);
        trainer.setMomentum(0.9f); 
        trainer.setOptimizer(OptimizerType.MOMENTUM); 
        trainer.train(imageSet);       
        
        // Test trained network
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.evaluate(neuralNet, imageSet);     
        System.out.println(evaluator);            
        
    }
        
    public static void main(String[] args) throws DeepNettsException, IOException {                                
            (new Cifar10Ce()).run();                
    }
}