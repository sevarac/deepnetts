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
import deepnetts.util.DeepNettsException;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.OptimizerType;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class JavaOneSponsors {
            
    int imageWidth = 128;
    int imageHeight = 128;
         
    String labelsFile = "datasets/JavaOneSponsors/labels.txt";
    String trainingFile = "datasets/JavaOneSponsors/train.txt";
   // String testFile = "datasets/JavaOneSponsors/test/test.txt";         
    
    static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());
    
    
    public void run() throws DeepNettsException, IOException {

        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        
        LOG.info("Loading images...");
        
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile), false);
    
        imageSet.invert();
        imageSet.zeroMean();
        imageSet.shuffle();        
        
        int labelsCount = imageSet.getLabelsCount();
        
        LOG.info("Done!");             
        LOG.info("Creating neural network...");

        // dodaj i bele slike kao negative u data set
        ConvolutionalNetwork javaOneNet = ConvolutionalNetwork.builder()
                                        .inputLayer(imageWidth, imageHeight, 1) 
                                        .convolutionalLayer(5, 5, 3, ActivationType.TANH)
                                        .maxPoolingLayer(2, 2, 2)                 
                                        .fullyConnectedLayer(30, ActivationType.TANH)
                                        .outputLayer(labelsCount, ActivationType.SOFTMAX)
                                        .lossFunction(CrossEntropyLoss.class)
                                        .randomSeed(123)
                                        .build();
                     
        LOG.info("Done!");       
        LOG.info("Training neural network"); 
        
           
        LOG.setLevel(Level.FINEST);
        // create a set of convolutional networks and do training, crossvalidation and performance evaluation
        BackpropagationTrainer trainer = new BackpropagationTrainer(javaOneNet);
        trainer.setLearningRate(0.01f)
               .setMomentum(0.7f)
               .setMaxError(0.6f)
               .setMaxIterations(100)
               .setOptimizer(OptimizerType.MOMENTUM);
        trainer.train(imageSet);   
          
        // Serialize network
        try {
            FileIO.writeToFile(javaOneNet, "javaonesponsors.dnet");
        } catch (IOException ex) {
            Logger.getLogger(JavaOneSponsors.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        // deserialize and evaluate neural network
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.evaluate(javaOneNet, imageSet);     
        System.out.println(evaluator);   
                          
//        BufferedImage image = ImageIO.read(new File("/home/zoran/Desktop/JavaOneSet/java/java1.jpg"));
//        DeepNettsImageClassifier imageClassifier = new DeepNettsImageClassifier(javaOneNet);
//        ClassificationResults<ClassificationResult> results = imageClassifier.classify(image);
//        System.out.println(results.toString());
       
                
    }
    
    
    
    
    public static void main(String[] args) {                     
            
        try {
            (new JavaOneSponsors()).run();
        } catch (DeepNettsException | IOException ex) {
            Logger.getLogger(JavaOneSponsors.class.getName()).log(Level.SEVERE, null, ex);
        }
   
                
    }
}