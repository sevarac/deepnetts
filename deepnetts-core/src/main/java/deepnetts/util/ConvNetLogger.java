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

import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.Tensor;
import java.util.Arrays;

/**
 * Methods for logging convolutional network (Logging utils)
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class ConvNetLogger { // this could be loggger or handler , and it should not be a singleton
    
    public boolean logInputs=false, logOutputs=true, logDeltas=true, logWeights=true, logDeltaWeights=true, logParams=false;
    
    private static ConvNetLogger instance = null;

    private ConvNetLogger() {  }
    
    public static ConvNetLogger getInstance() {
        if (instance == null)  {
            instance = new ConvNetLogger();
        }
        
        return instance;
    }
    
    
    
  
    public String logNetwork(NeuralNetwork convNet) {
        StringBuilder sb = new StringBuilder();
        
        int layerIdx=0;
        
        for(AbstractLayer layer : convNet.getLayers()) {
            sb.append(logLayer(layer, layerIdx));            
            layerIdx++;
        }
                
        return sb.toString();
    }
    
    public String logLayer(ConvolutionalNetwork convNet, int layerIdx) {
        return logLayer(convNet.getLayers().get(layerIdx), layerIdx);
    }
            
    private String logLayer(AbstractLayer layer, int layerIdx) {
        StringBuilder sb = new StringBuilder();
           if (layer instanceof InputLayer) {
                 sb.append("{layer:InputLayer");
                 if (logParams) sb.append(", layerIdx:"+layerIdx+", width:"+layer.getWidth()+", height:"+layer.getHeight()+", depth:"+layer.getDepth());
                 if (logOutputs) sb.append(", outputs:").append(layer.getOutputs());                 
                 sb.append("}").append(System.lineSeparator());                   
            } else if (layer instanceof ConvolutionalLayer) {
                sb.append("{layer:ConvolutionalLayer,");
                if (logOutputs) sb.append(System.lineSeparator()).append(" outputs:").append(layer.getOutputs()).append(System.lineSeparator());  ;                 
                if (logWeights) {
                   Tensor[] filters = ((ConvolutionalLayer) layer).getFilters();
                   sb.append(" filters:[").append(Arrays.toString(filters));
               }
               if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" biases:").append(Arrays.toString(layer.getBiases()));                   
               if (logDeltas) sb.append(", ").append(System.lineSeparator()).append(" deltas:").append(layer.getDeltas());
                if (logDeltaWeights) { 
                    Tensor[] deltaWeights = ((ConvolutionalLayer) layer).getFilterDeltaWeights();
                    sb.append(System.lineSeparator()).append(" deltaWeights:").append(Arrays.toString(deltaWeights));
                    sb.append(System.lineSeparator()).append(" delta biasess:").append(Arrays.toString(layer.getDeltaBiases()));
                } 
                sb.append("}").append(System.lineSeparator());
            } else if (layer instanceof FullyConnectedLayer) {
                sb.append("{layer:FullyConnectedLayer");
                if (logParams) sb.append(", layerIdx:"+layerIdx+", depth:"+layer.getDepth());
                if (logOutputs) sb.append(", ").append(System.lineSeparator()).append(" outputs:").append(layer.getOutputs());                 
                if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" weights:").append(layer.getWeights());                 
                if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" biases:").append(Arrays.toString(layer.getBiases()));                 
                if (logDeltas) sb.append(System.lineSeparator()).append(" deltas:").append(layer.getDeltas());
                if (logDeltaWeights) { 
                    sb.append(System.lineSeparator()).append(" deltaWeights:").append(layer.getDeltaWeight());
                    sb.append(System.lineSeparator()).append(" delta biasess:").append(Arrays.toString(layer.getDeltaBiases()));
                }
                sb.append("}").append(System.lineSeparator());                
            } else if (layer instanceof SoftmaxOutputLayer) {
                sb.append("{layer:SoftmaxOutputLayer");
                if (logParams)  sb.append(", layerIdx:"+layerIdx+", depth:"+layer.getDepth());
                if (logOutputs) sb.append(", ").append(System.lineSeparator()).append(" outputs:").append(layer.getOutputs());                 
                if (logOutputs) sb.append(", ").append(System.lineSeparator()).append(" errors:").append(Arrays.toString(((SoftmaxOutputLayer) layer).getOutputErrors()));                                 
                if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" weights:").append(layer.getWeights());                 
                if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" biases:").append(Arrays.toString(layer.getBiases()));                                                 
                if (logDeltas)  sb.append(", ").append(System.lineSeparator()).append(" deltas:").append(layer.getDeltas());
                if (logDeltaWeights) { 
                    sb.append(System.lineSeparator()).append(" deltaWeights:").append(layer.getDeltaWeight());
                    sb.append(System.lineSeparator()).append(" delta biasess:").append(Arrays.toString(layer.getDeltaBiases()));
                }                              
                sb.append("}").append(System.lineSeparator());
            } else if (layer instanceof OutputLayer) {
                sb.append("{layer:SigmoidOutputLayer");
                if (logParams) sb.append(", layerIdx:"+layerIdx+", depth:"+layer.getDepth());
                if (logOutputs) sb.append(", ").append(System.lineSeparator()).append(" outputs:").append(layer.getOutputs());                 
                if (logOutputs) sb.append(", ").append(System.lineSeparator()).append(" errors:").append(Arrays.toString(((OutputLayer) layer).getOutputErrors()));     
                if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" weights:").append(layer.getWeights());                 
                if (logWeights) sb.append(", ").append(System.lineSeparator()).append(" biases:").append(Arrays.toString(layer.getBiases()));                                 
                if (logDeltas)  sb.append(", ").append(System.lineSeparator()).append(" deltas:").append(layer.getDeltas());
                if (logDeltaWeights) { 
                    sb.append(System.lineSeparator()).append(" deltaWeights:").append(layer.getDeltaWeight());
                    sb.append(System.lineSeparator()).append(" delta biasess:").append(Arrays.toString(layer.getDeltaBiases()));
                }                              
                sb.append("}").append(System.lineSeparator());
            }
           return sb.toString();
    }

    public void logInputs(boolean logInputs) {
        this.logInputs = logInputs;
    }

    public void logOutputs(boolean logOutputs) {
        this.logOutputs = logOutputs;
    }

    public void logDeltas(boolean logDeltas) {
        this.logDeltas = logDeltas;
    }

    public void logWeights(boolean logWeights) {
        this.logWeights = logWeights;
    }

    public void logDeltaWeights(boolean logDeltaWeights) {
        this.logDeltaWeights = logDeltaWeights;
    }

    public void logParams(boolean logParams) {
        this.logParams = logParams;
    }

    public boolean logInputs() {
        return logInputs;
    }

    public boolean logOutputs() {
        return logOutputs;
    }

    public boolean logDeltas() {
        return logDeltas;
    }

    public boolean logWeights() {
        return logWeights;
    }

    public boolean logDeltaWeights() {
        return logDeltaWeights;
    }

    public boolean logParams() {
        return logParams;
    }
                           
}
