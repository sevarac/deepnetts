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

import java.io.Serializable;
import java.util.Arrays;

/**
 * This class represents multidimensional array/matrix (can be 1D, 2D, 3D or 4D).
 * 
 * @author Zoran Sevarac
 */
public class Tensor implements Serializable {
    
    // tensor dimensions, could be an array to get n-dim // use int array as a shape
    private final int cols, rows, depth, fourthDim, dimensions;
    
    /**
     * Values stored in this tensor
     *  make it final , only input layer and tests sets values
     */
    private float values[]; // todo: use ByteBuffer instead of array in order to avoid range checking
    
    /**
     * Creates a single row tensor with specified values.
     * 
     * @param values values of column tensor
     */
    public Tensor(final float... values) {
        this.rows = 1;
        this.cols = values.length;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 1;
        this.values = values;
    }    

    /**
     * Creates a 2D tensor / matrix with specified values.
     * 
     * @param vals 
     */    
    public Tensor(final float[][] vals) {
        this.rows = vals.length; 
        this.cols = vals[0].length;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 2;        
        this.values = new float[rows*cols];

        // copy array to single dim
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                set(row, col, vals[row][col]);
            }
        }        
    }  
    
    /**
     * Creates a 3D tensor from specified 3D array
     * 
     * @param vals 2D array of tensor values
     */
    public Tensor(final float[][][] vals) {        
        this.depth = vals.length;        
        this.rows = vals[0].length; 
        this.cols = vals[0][0].length;
        

        this.fourthDim = 1;
        this.dimensions = 3;        
        this.values = new float[rows*cols*depth];

        // copy array
        for (int z = 0; z < depth; z++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    set(row, col, z, vals[z][row][col]);
                }
            }
        }
    }    
    
    public Tensor(final float[][][][] vals) {        
        
        this.fourthDim = vals.length;        
        this.depth = vals[0].length;        
        this.rows = vals[0][0].length; 
        this.cols = vals[0][0][0].length;
        
        this.dimensions = 4;        
        this.values = new float[rows*cols*depth*fourthDim];

        // copy array
        for (int f = 0; f < fourthDim; f++) {
            for (int z = 0; z < depth; z++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        set(row, col, z, f, vals[f][z][row][col]);
                    }
                }
            }
        }
    }
        
   
    /**
     * Creates an empty single row tensor with specified number of columns.
     * 
     * @param cols number of columns
     */
    public Tensor(int cols) {        
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: "+cols);        
        
        this.cols = cols;
        this.rows = 1;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 1;
        values = new float[cols];        
    }    
    
    public Tensor(int cols, float val) {        
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: "+cols);        
        
        this.cols = cols;
        this.rows = 1;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 1;
        values = new float[cols];    
        
        for(int i=0; i<values.length; i++) {
            values[i] = val;
        }
    }       
    
    /**
     * Creates a tensor with specified number of rows and columns.
     * 
     * @param rows number of rows
     * @param cols number of columns
     */
    public Tensor(int rows, int cols) {        
        if (rows < 0) throw new IllegalArgumentException("Number of rows cannot be negative: "+rows);
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: "+cols);
        
        this.rows = rows;
        this.cols = cols;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 2;
        values = new float[rows * cols];        
    }
    
    
    public Tensor(int rows, int cols, float[] values) {    
        if (rows < 0) throw new IllegalArgumentException("Number of rows cannot be negative: "+rows);
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: "+cols);
        if (rows*cols != values.length) throw new IllegalArgumentException("Number of values does not match tensor dimensions! " + values.length);        
        
        this.rows = rows;
        this.cols = cols;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 2;
                
        this.values = values;        
    }    
    
    /**
     * Creates a 3D tensor with specified number of rows, cols and depth.
     * 
     * @param rows number of rows
     * @param cols number of columns
     * @param depth tensor depth
     */
    public Tensor(int rows, int cols, int depth) {
        if (rows < 0) throw new IllegalArgumentException("Number of rows cannot be negative: "+rows);
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: "+cols);
        if (depth < 0) throw new IllegalArgumentException("Depth cannot be negative: "+depth);
        
        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        this.fourthDim = 1;
        this.dimensions = 3;
        this.values = new float[rows * cols * depth];
    }    

    // cols, rows, 3rd, 4th?
    public Tensor(int rows, int cols, int depth, int fourthDim) {
        if (rows < 0) throw new IllegalArgumentException("Number of rows cannot be negative: "+rows);
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: "+cols);
        if (depth < 0) throw new IllegalArgumentException("Depth cannot be negative: "+depth);
        if (fourthDim < 0) throw new IllegalArgumentException("fourthDim cannot be negative: "+fourthDim);
        
        this.rows = rows;
        this.cols = cols;
        this.depth = depth;       
        this.fourthDim = fourthDim;
        this.dimensions = 4;
        this.values = new float[rows * cols * depth * fourthDim];
    }        
    
    public Tensor(int rows, int cols, int depth, float[] values) {
        if (rows < 0) throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);        
        if (cols < 0) throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);        
        if (depth < 0) throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        if (rows*cols*depth != values.length) throw new IllegalArgumentException("Number of values does not match tensor dimensions! " + values.length);
        
        this.cols = cols;
        this.rows = rows;
        this.depth = depth;
        this.fourthDim = 1;
        this.dimensions = 3;
        this.values = values;
    }
    
    private Tensor(Tensor t) {
        this.cols = t.cols;
        this.rows = t.rows;
        this.depth = t.depth;
        this.fourthDim = t.fourthDim;
        this.dimensions = t.dimensions;
        values = new float[t.values.length];
        
        System.arraycopy(t.values, 0, values, 0, t.values.length);
    }
    
    
    /**
     * Gets value at specified index position.
     * 
     * @param idx
     * @return 
     */
    public final float get(final int idx) {
        return values[idx];
    }
    
    /**
     * Sets value at specified index position.
     * 
     * @param idx
     * @param val
     * @return 
     */
    public final float set(final int idx, final float val) {
        return values[idx] = val;
    }    
    
    // make sure this method gets inlined - final?   keeping hot methods small (35 bytecodes or less) final migh help, it will get inlined if its a hotspot - frequent calls
   /**
    * Returns matrix value at row, col
    * 
    * @param col 
    * @param row 
    * @return value at [row, col]
    */
    public final float get(final int row, final int col) {
        final int idx = row * cols + col;
        return values[idx];
    }
        
    
    /**
     * Sets matrix value at specified [row, col] position
     * 
     * @param row matrix roe
     * @param col matrix col
     * @param val value to set
     */
    public final void set(final int row, final int col, final float val) {
        final int idx = row * cols + col;
        values[idx] = val;
    }
        
    /**
     * Returns value at row, col, z
     * 
     * @param col
     * @param row
     * @param z
     * @return 
     */
    public final float get(final int row, final int col, final int z) {               
        final int idx = z * cols * rows + row * cols + col;
        return values[idx];
    }  
        
    public final void set(final int row, final int col,  final int z, final float val) {
        final int idx = z * cols * rows + row * cols + col;
        values[idx] = val;
    }    
    
    public final float get(final int row, final int col, final int z, final int fourth) {               
        final int idx = fourth * rows * cols * depth + z * rows * cols  + row * cols + col;
        return values[idx];
    }  
    
    public final void set(final int row, final int col, final int z, final int fourth, final float val) {
        final int idx = fourth * rows * cols * depth + z * rows * cols  + row * cols + col;
        values[idx] = val;
    }    
        
    public final float[] getValues() {
        return values;
    }

    public final void setValues(float... values) {     
        this.values = values;
    }
      
    public final int getCols() {
        return cols;
    }
    
    public final int getRows() {
        return rows;
    }

    public final int getDepth() {
        return depth;
    }

    public final int getFourthDim() {
        return fourthDim;
    }
    
    public final int getDimensions() {
        return dimensions;
    }
                    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        // toString logging
        if (ConvNetLogger.getInstance().logParams()) {
            sb.append("{Tensor, cols:"+cols+", rows:"+rows+", depth:"+depth+", fourthDim:"+fourthDim+", dimensions:"+dimensions);
            sb.append(",values:");
        }
        
        sb.append("[");
        for (int i = 0; i < values.length; i++) {
            sb.append(values[i]);
            if (i<values.length-1) sb.append(", ");
        }
        sb.append("]");

        // maybe use Arrays.toString(values)

        return sb.toString();
    }
   

    /**
     * Adds specified value to matrix value at position x, y
     * 
     * @param col
     * @param row
     * @param value 
     */
    public final void add(final int row, final int col, final float value) {
         final int idx = row * cols + col;
         values[idx] += value;
    }
    
    public void add(final int row, final int col, final int z, final float value) {
         final int idx = z * cols * rows + row * cols + col;
         values[idx] += value;
    }    
    
    public final void add(final int row, final int col, final int z, final int fourth, final float value) {
        final int idx = fourth * cols * rows * depth + z * cols * rows + row * cols + col;
        values[idx] += value;
    }        
    
    public final void add(final int idx, final float value) {
        values[idx] += value;
    }        
    
    /**
     * Adds specified tensor t to this tensor.
     * 
     * @param t tensor to add
     */
    public final void add(Tensor t) {
        for(int i=0; i<values.length; i++) {
            values[i] += t.values[i];
        }
    }    
        
    public final void sub(final int row, final int col, final float value) {
         final int idx = row * cols + col;
         values[idx] -= value;
    }
    
    public final void sub(final int row, final int col, final int z, final float value) {
         final int idx = z * rows * cols + row * cols + col;
         values[idx] -= value;
    }    
    
    public final void sub(final int row, final int col, final int z, final int fourth, final float value) {
        final int idx = fourth * rows * cols * depth + z * rows * cols  + row * cols + col;
        values[idx] -= value;
    }        
         
    public final void sub(final Tensor t) {
        for(int i=0; i < values.length; i++) {
            values[i] -= t.values[i];
        }
    }
    
    /**
     * Subtracts m2 from m1
     * 
     * @param t1
     * @param t2 
     */
    public final static void sub(final Tensor t1, final Tensor t2) {
        for(int i=0; i < t1.values.length; i++) {
            t1.values[i] -= t2.values[i];            
        }
    }
         
    public final void div(final float value) {
        for(int i=0; i<values.length; i++) {
            values[i] /= value;
        }
    }        
         
    public final void fill(final float value) {
        for (int i = 0; i < values.length; i++) {
            values[i] = value;
        }
    }
    
    public static final void fill(final float[] array, final float val) {
        for (int i = 0; i < array.length; i++) 
            array[i] = val;
    }    
    
    public static void div(final float[] array, final float val) {
        for(int i = 0; i < array.length; i++)
            array[i] /= val;        
    }

    public static final void sub(final float[] array1, final float[] array2) {
        for(int i = 0; i < array1.length; i++)
            array1[i] -= array2[i];   
    }    
    
    public static final void add(final float[] array1, final float[] array2) {
        for(int i = 0; i < array1.length; i++)
            array1[i] += array2[i];   
    }        
    
    public static final void copy(final Tensor src, final Tensor dest) {
        System.arraycopy(src.values, 0, dest.values, 0, src.values.length);
    }
    
    public static final void copy(final float[] src, final float[] dest) {
        System.arraycopy(src, 0, dest, 0, src.length);
    }    
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Tensor other = (Tensor) obj;
        if (this.cols != other.cols) {
            return false;
        }
        if (this.rows != other.rows) {
            return false;
        }
        if (this.depth != other.depth) {
            return false;
        }
        if (this.fourthDim != other.fourthDim) {
            return false;
        }
        if (this.dimensions != other.dimensions) {
            return false;
        }
        if (!Arrays.equals(this.values, other.values)) {
            return false;
        }
        return true;
    }

    public boolean equals(Tensor t2, float delta) {
        float[] arr2 = t2.getValues();
        
        for(int i=0; i < values.length; i++ ) {
            if (Math.abs(values[i]-arr2[i]) > delta) return false;
        }
        return true;
    }    
    
        
    // add clone using apache clone builder
    
    public static Tensor zeros(int cols) {
        return new Tensor(cols, 0f);
    }
    
    public static Tensor ones(int cols) {
        return new Tensor(cols, 1f);
    }    
            
}
