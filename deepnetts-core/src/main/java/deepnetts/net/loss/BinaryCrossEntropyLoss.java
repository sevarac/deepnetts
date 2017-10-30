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
    
package deepnetts.net.loss;

import deepnetts.util.DeepNettsException;
import deepnetts.net.NeuralNetwork;
import java.io.Serializable;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class BinaryCrossEntropyLoss implements LossFunction, Serializable {
    private final float[] outputError;
    private float[] actualOutput;
    private float[] targetOutput;
    
    // ovi treba da racunaju i ukupnu gresku i da vracaju total error a ne u Backrpop traineru. Da li treba 1/n i ovde? Verovatno, mada mozda i ne
  
    public BinaryCrossEntropyLoss(NeuralNetwork convNet) {
        if (convNet.getOutputLayer().getWidth()>1) throw new DeepNettsException("BinaryCrossEntropyLoss can be only used with networks with single output!");
        
        outputError = new float[1];
    }
       
    /**
     * Calculates and returns error vector for specified actual and target outputs.
     * 
     * @param actualOutput actual output from the neural network
     * @param targetOutput target/desired output of the neural network
     * @return error vector for specified actual and target outputs
     */
    @Override
    public float[] calculateOutputError(final float[] actualOutput,  final float[] targetOutput) {
        this.actualOutput = actualOutput;    
        this.targetOutput = targetOutput;
        
        outputError[0] = actualOutput[0] - targetOutput[0]; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona - ovo se koristi za deltu izlaznog neurona              
        return outputError;        
    }
    
    /**
     * Return the total error for the current pattern 
     * 
     * @return 
     */
    @Override
    public float getPatternError() {
        return (float)-(targetOutput[0]*Math.log(actualOutput[0]) + (1-targetOutput[0])*Math.log(1-actualOutput[0]));
    }
    
}
