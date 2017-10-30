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
    
package deepnetts.data;

import deepnetts.util.Tensor;
import java.util.Arrays;

/**
 * Represents an item in a supervised data set. 
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class BasicDataSetItem  implements DataSetItem {

    private final Tensor input; // network input
    private final float[] targetOutput; // for classifiers target can be index, int 
        
    public BasicDataSetItem(float[] in, float[] targetOutput) {
        this.input = new Tensor(in);
        this.targetOutput = targetOutput;
    }    
    
    public BasicDataSetItem(Tensor input, float[] targetOutput) {
        this.input = input;
        this.targetOutput = targetOutput;
    }
           
    @Override
    public Tensor getInput() {
        return input;
    }

    @Override
    public float[] getTargetOutput() {
        return targetOutput;
    }
    
    public int size() {
        return input.getCols();
    }

    @Override
    public String toString() {
        return "VectorDataItem{" + "input=" + input + ", targetOutput=" + Arrays.toString(targetOutput) + '}';
    }
        
}