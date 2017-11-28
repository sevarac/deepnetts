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

import deepnetts.net.NeuralNetwork;
import java.io.Serializable;

/**
 * Mean Squared Error Loss function
 * 
 * Should i provide total error from here?
 * 
 * TODO: count patterns and return mean from here. makes more sense
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public final class MeanSquaredErrorLoss implements LossFunction, Serializable {

    private final float[] outputError;
    private float totalPatternError;
    private float totalError;
    //private int patternCount=0;
        

    public MeanSquaredErrorLoss(NeuralNetwork convNet) {
        outputError = new float[convNet.getOutputLayer().getWidth()];
    }

    /**
     * Returns output error vector and adds it to total error.
     * 
     * @param actualOutput
     * @param targetOutput
     * @return 
     */
    @Override
    public float[] calculateOutputError(final float[] actualOutput, final float[] targetOutput) {
        totalPatternError = 0;
        for (int i = 0; i < actualOutput.length; i++) {
            outputError[i] = actualOutput[i] - targetOutput[i];
            totalPatternError += outputError[i] * outputError[i];
        }

        totalError += 0.5 * totalPatternError;
        
        return outputError;
    }

    @Override
    public float getPatternError() {
        return totalPatternError;
    }
    
    public float getTotalError() {
        return totalError;
    }
    
    public void resetTotalError() {
        totalError = 0;
        // patternCount=0;
    }
    

}