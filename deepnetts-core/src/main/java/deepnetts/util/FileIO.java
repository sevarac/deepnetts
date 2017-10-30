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
    
package deepnetts.util;

import deepnetts.net.layers.ActivationType;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.LayerType;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * File utilities for saving and loading neural networks.
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class FileIO {

    /**
     * This class has only static utility methods so we don't need instances
     */
    private FileIO() { }
    
    /**
     * Serializes specified neural network to file with specified file.
     * 
     * @param neuralNet neural network to save
     * @param fileName name of the file
     * @throws IOException if something goes wrong
     */
    public static void writeToFile(NeuralNetwork neuralNet, String fileName) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName))) {
            oos.writeObject(neuralNet);
        }
    }
    
    public static void writeToFileAsJson(ConvolutionalNetwork convNet, String fileName) throws IOException {
        String jsonStr = toJson(convNet);
        try (PrintWriter pw = new PrintWriter(new File(fileName))) {
            pw.print(jsonStr);
        }
    }    
    
    public static <T> T createFromFile(String fileName, Class<T> clazz) throws IOException, ClassNotFoundException {
        T neuralNet;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName))) {  
            neuralNet = clazz.cast(ois.readObject()) ;                   
        }
        return neuralNet;
    }
    
    public static NeuralNetwork createFromFile(File file) throws IOException, ClassNotFoundException {
        NeuralNetwork nnet;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            nnet = (ConvolutionalNetwork) ois.readObject();        
        }
        return nnet;
    }    
    
    public static String  toJson(ConvolutionalNetwork convNet) {
        JSONObject convNetJson = new JSONObject();
        JSONArray layers = new JSONArray();
        
        InputLayer inputLayer= convNet.getInputLayer();
                
        JSONObject inputLayerJson = new JSONObject();
        inputLayerJson.put("layerType", LayerType.INPUT.toString());
        inputLayerJson.put("width", inputLayer.getWidth());
        inputLayerJson.put("height", inputLayer.getHeight());
        inputLayerJson.put("channels", inputLayer.getDepth());                
        layers.put(inputLayerJson);             
        
        for(AbstractLayer layer : convNet.getLayers()){
            if (layer instanceof ConvolutionalLayer) {
                ConvolutionalLayer convLayer = (ConvolutionalLayer)layer;
                JSONObject convLayerJson = new JSONObject();
                convLayerJson.put("layerType", LayerType.CONVOLUTIONAL);
                convLayerJson.put("filterWidth", convLayer.getFilterWidth());
                convLayerJson.put("filterHeight", convLayer.getFilterHeight());
                convLayerJson.put("channels", convLayer.getDepth()); // channels
                convLayerJson.put("stride", convLayer.getStride());
                convLayerJson.put("activation", convLayer.getActivationType());     
                layers.put(convLayerJson);   
            } else if (layer instanceof MaxPoolingLayer) {
                MaxPoolingLayer maxPooling= (MaxPoolingLayer)layer;
                JSONObject poolLayerJson = new JSONObject();
                poolLayerJson.put("layerType", LayerType.MAXPOOLING);
                poolLayerJson.put("filterWidth", maxPooling.getFilterWidth());
                poolLayerJson.put("filterHeight", maxPooling.getFilterHeight());
                poolLayerJson.put("stride", maxPooling.getStride());    
                layers.put(poolLayerJson);  
            } else if (layer instanceof FullyConnectedLayer) {
                JSONObject fullyConnLayerJson = new JSONObject();
                fullyConnLayerJson.put("layerType", LayerType.FULLYCONNECTED);
                fullyConnLayerJson.put("width", layer.getWidth());
                fullyConnLayerJson.put("activation", layer.getActivationType());
                layers.put(fullyConnLayerJson);   
            } else if (layer instanceof SoftmaxOutputLayer) {
                JSONObject outputLayerJson = new JSONObject();
                outputLayerJson.put("layerType", LayerType.OUTPUT);
                outputLayerJson.put("width", layer.getWidth());
                outputLayerJson.put("activation", layer.getActivationType());
                layers.put(outputLayerJson);  
            } else if (layer instanceof OutputLayer) {
                JSONObject outputLayerJson = new JSONObject();
                outputLayerJson.put("layerType", LayerType.OUTPUT);
                outputLayerJson.put("width", layer.getWidth());
                outputLayerJson.put("activation", layer.getActivationType());
                layers.put(outputLayerJson);  
            }            
        }
        
        convNetJson.put("layers", layers);

        return convNetJson.toString();
    }
    
     public static ConvolutionalNetwork createFromJson(String jsonStr) {
        JSONObject obj = new JSONObject(jsonStr);
        return createFromJson(obj);
     }
     
     public static ConvolutionalNetwork createFromJson(File file) throws FileNotFoundException, IOException {
         BufferedReader br = new BufferedReader(new FileReader(file));
         StringBuilder sb = new StringBuilder();
         String line;
         while((line = br.readLine()) != null) {
             sb.append(line).append(System.lineSeparator());
         }
         return createFromJson(sb.toString());
     }
    
    public static ConvolutionalNetwork createFromJson(JSONObject jsonObj) {
        JSONArray jsonLayers = jsonObj.getJSONArray("layers");
        // dodaj loss funkciju i vidi da li imaju jos neki parametri /slucajevi koje sam propustio
        
        ConvolutionalNetwork.Builder convNetBuilder = new ConvolutionalNetwork.Builder();
        
        
        int width, height, channels, filterWidth, filterHeight, stride, padding;
        String activation;
        
        // random seed, loss 
        for(Object jsonLayerObject : jsonLayers) {
            JSONObject layerObj = (JSONObject)jsonLayerObject;
            
            switch(LayerType.valueOf( layerObj.getString("layerType").toUpperCase() ) ) {
                case INPUT :
                        width = layerObj.getInt("width");
                        height = layerObj.getInt("height");
                        channels = layerObj.getInt("channels");
                        convNetBuilder.inputLayer(width, height, channels);                    
                break;
                case CONVOLUTIONAL :
                        filterWidth = layerObj.getInt("filterWidth");
                        filterHeight = layerObj.getInt("filterHeight");
                        stride = layerObj.getInt("stride");
                        channels = layerObj.getInt("channels");   
                        activation = layerObj.getString("activation").toUpperCase();      
                        convNetBuilder.convolutionalLayer(filterWidth, filterHeight, channels, ActivationType.valueOf(activation));                    
                break;
                case MAXPOOLING :
                        filterWidth = layerObj.getInt("filterWidth");
                        filterHeight = layerObj.getInt("filterHeight");
                        stride = layerObj.getInt("stride");
                        convNetBuilder.maxPoolingLayer(filterWidth, filterHeight, stride);                    
                break;
                case FULLYCONNECTED :
                        width = layerObj.getInt("width");
                        activation = layerObj.getString("activation").toUpperCase();                          
                        convNetBuilder.fullyConnectedLayer(width, ActivationType.valueOf(activation));                            
                break;                
                case OUTPUT :
                        width = layerObj.getInt("width");
                        activation = layerObj.getString("activation").toUpperCase();
                                              
                        if (activation.equals(ActivationType.SIGMOID.toString())) {
                            convNetBuilder.outputLayer(width, OutputLayer.class);                  
                            convNetBuilder.lossFunction(MeanSquaredErrorLoss.class);
                        } else if (activation.equals(ActivationType.SOFTMAX.toString())) {          
                            convNetBuilder.outputLayer(width, SoftmaxOutputLayer.class);        
                            convNetBuilder.lossFunction(CrossEntropyLoss.class);
                        }                        
                break;            
            }            
        }
        
        return convNetBuilder.build();
    }    
}
