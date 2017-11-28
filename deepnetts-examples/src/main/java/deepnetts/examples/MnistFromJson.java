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
import deepnetts.util.FileIO;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


/**
 * Example of training Convolutional network for MNIST data set.
 * Note: in order to run this example you must download mnist data set and update image paths in train.txt file
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MnistFromJson {
            
    int imageWidth  = 28;
    int imageHeight = 28;
         
    String labelsFile   = "datasets/mnist/labels.txt";
    String trainingFile = "datasets/mnist/train.txt";
    
    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());        
    
    public void run() throws DeepNettsException, IOException {
        
        LOGGER.info("Training convolutional network with MNIST data set");        
        LOGGER.info("Loading images...");

        
        // create a data set from images and labels
        // load data set properties
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);        
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile), true);
        imageSet.invert();
        imageSet.zeroMean();
        imageSet.shuffle();
        
        // koad data / images from properties file
        
        int labelsCount = imageSet.getLabelsCount();
                  
        LOGGER.info("Creating neural network...");
                    
         // create convolutional neural network architecture from json file 
        ConvolutionalNetwork neuralNet = FileIO.createFromJson(new File("mnistDemo.json"), ConvolutionalNetwork.class);   
                           
        LOGGER.info("Training neural network"); 
        
        // load training properties from properties file 
        Properties trainProp = new Properties();
        trainProp.load(new FileReader("training.properties"));        
        
        // create a trainer and train network
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet, trainProp); // send props as a param to set all peoperties from properties file        
        
        trainer.setOptimizer(OptimizerType.MOMENTUM);
        
        trainer.train(imageSet);   
                       
        // Test trained network
        ClassifierEvaluator tester = new ClassifierEvaluator();
        tester.evaluate(neuralNet, imageSet);     
        System.out.println(tester);                          
                
        // Save network to file as json
        //FileIO.writeToFile(neuralNet, "mnistDemo.dnet");
        FileIO.writeToFileAsJson(neuralNet, "mnistDemo2.json");
    }
      
        
    public static void main(String[] args) throws IOException {                             
            (new MnistFromJson()).run();                   
    }
}