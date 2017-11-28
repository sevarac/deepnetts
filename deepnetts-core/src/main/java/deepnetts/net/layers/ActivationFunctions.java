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
    
package deepnetts.net.layers;

import deepnetts.util.DeepNettsException;

/**
 * Typical mathematic functions used as layer activation functions in neural networks.
 * 
 * TODO: add slope and amplitude for sigmoid, tanh etc.
 * annottations o automaticly generate enums for activation types?
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public final class ActivationFunctions {
       
    /**
     * Private constructor to prevent instantiation of this class (only static methods)
     */
    private  ActivationFunctions() {  }
        
    
    /**
     * Returns the result of the specified function for specified input.
     * 
     * @param type
     * @param x
     * @return 
     */
    public static final float calc(final ActivationType type, final float x) {

        switch(type) {
            case SIGMOID:
                 return sigmoid(x);
                
            case TANH:
                return tanh(x);
                
            case RELU:
                return relu(x);    
                
            case LINEAR:
                return linear(x);
        }
        
        throw new DeepNettsException("Unknown transfer function type!");
    }
    
    public static final float prime(ActivationType type, float y) {
        
        switch(type) {
            case SIGMOID:
                 return sigmoidPrime(y);
                
            case TANH: 
                return tanhPrime(y);
                
            case RELU:
                return reluPrime(y);                
                
            case LINEAR:
                return linearPrime(y);                
        }
        
        throw new DeepNettsException("Unknown transfer function type!");
    }    
    
    
    /**
     * Basic Sigmoid Function on interval (0..1). 
     *       
     *  y = 1 / (1+e^(-x))
     * 
     * TODO: slope, amplitude, translate? (all these params could be trainable...)
     *       maybe add annotations so Type enums can be generated automatically
     * 
     * @param x
     * @return value of sigmoid function calculated for input x
     */
    // @Activation("sigmoid")
    public static final float sigmoid(final float x) {
       return 1 / (1 + (float)Math.exp(-x));  
    } 
    
    /**
     * First derivative of sigmoid function .
     * 
     *  f1 = y(1-y)
     * 
     * @param y sigmoid function output
     * @return first derivative of sigmoid
     */
    // @ActivationPrime("sigmoid")
    public static final float sigmoidPrime(final float y) {
       return y*(1-y);
    } 
            
    /**
     * Tanh function (sigmoid on interval (-1, 1)).
     * 
     *  y = ((e^2x)-1 ) / ((e^2x)+1)
     * 
     * TODO: amplitude, slope, +ax kako bi u izvodu imao +a za flatspot
     * 
     * @param x
     * @return value of tanh function calculated for input x
     */
    public static final float tanh(final float x) {
       // x = x*2/3;
       // float a=1.7159;
       final float e2x = (float)Math.exp(2*x);   
       return (e2x-1) / (e2x+1); // calculate tanh 
    } 
    
    /**
     * First derivative of tanh function.
     * 
     * @param y
     * @return 
     */
    public static final float tanhPrime(final float y) {     
        return (1-y*y);
    }
        
    public static final float relu(final float x) {
       return Math.max(0, x);  
    }     
    
    public static final float reluPrime(final float y) {
       return ( y > 0 ? 1 : 0);
    }         
    
    /**
     * Linear activation function.
     * y = x
     * 
     * TODO: y = kx + n (with k and n trainable)
     * 
     * @param x
     * @return 
     */
    public static final float linear(final float x) {
       return x;  
    }     
    
    /**
     * First derivative of linear function.
     * For y = x, derivative is allways 1
     * 
     * @param y
     * @return 
     */
    public static final float linearPrime(final float y) {
       return 1;
    }     
        
}