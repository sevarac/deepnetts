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

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import visrec.classifier.ClassificationResult;
import visrec.classifier.ClassificationResults;

/**
 * This example shows how to load and create instance of trained network from file.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LoadAndUseTrainedNetwork {
    
    public static void main(String[] args) {

        try {
            ConvolutionalNetwork neuralNet =  FileIO.createFromFile("javaOneSponsors.net", ConvolutionalNetwork.class);

            DeepNettsImageClassifier imageClassifier = new DeepNettsImageClassifier(neuralNet);    // this image recognize shoul dbe used from visrec api
            ClassificationResults<ClassificationResult> results = imageClassifier.classify(new File("/home/zoran/Desktop/JavaOneSponsors/redhat.png"));
            System.out.println(results.toString());

        } catch (IOException | ClassNotFoundException ioe) {
            Logger.getLogger(LoadAndUseTrainedNetwork.class.getName()).log(Level.SEVERE, null, ioe);
        }
     
    }    
}
