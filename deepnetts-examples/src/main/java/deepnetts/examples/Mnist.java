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
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.OptimizerType;
import deepnetts.util.DeepNettsException;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;


/**
 * Example of training Convolutional network for MNIST data set.
 * Note: in order to run this example you must download mnist data set and update image paths in train.txt file
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class Mnist {
            
    int imageWidth  = 28;
    int imageHeight = 28;
         
    String labelsFile   = "datasets/mnist/labels.txt";
    String trainingFile = "datasets/mnist/train.txt";
    
    static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());
    
    
    public void run() throws DeepNettsException, IOException {
        
        LOG.info("Training convolutional network with MNIST data set");        
        LOG.info("Loading images...");
       
        // create a data set from images and labels
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);        
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile), true);
        imageSet.invert();
        imageSet.zeroMean();
        imageSet.shuffle();
                
        int labelsCount = imageSet.getLabelsCount();
        
        LOG.info("Done!");             
        LOG.info("Creating neural network...");
                    
         // create convolutional neural network architecture           
        ConvolutionalNetwork neuralNet = new ConvolutionalNetwork.Builder()
                                        .inputLayer(imageWidth, imageHeight)
                                        .convolutionalLayer(5, 3)
                                        .maxPoolingLayer(2, 2)        
                                        .convolutionalLayer(3, 6) 
                                        .maxPoolingLayer(2, 2)       
                                        .fullyConnectedLayer(30)
                                        .fullyConnectedLayer(20)
                                        .outputLayer(labelsCount, ActivationType.SOFTMAX)
                                        .activationFunction(ActivationType.RELU) 
                                        .lossFunction(CrossEntropyLoss.class)
                                        .randomSeed(123)       
                                        .build();   
                
        LOG.info("Done!");       
        LOG.info("Training neural network"); 
                 
        // create a trainer and train network
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setLearningRate(0.03f)
                .setMomentum(0.7f)
                .setMaxError(0.02f)
                .setBatchMode(false)
                .setOptimizer(OptimizerType.MOMENTUM);
        trainer.train(imageSet);   
                       
        // Test trained network
        ClassifierEvaluator tester = new ClassifierEvaluator();
        tester.evaluate(neuralNet, imageSet);     
        System.out.println(tester);                          
                
        // Save network to file as json
        FileIO.writeToFile(neuralNet, "mnistDemo.dnet");
    }
      
        
    public static void main(String[] args) throws IOException {                             
            (new Mnist()).run();                   
    }
}