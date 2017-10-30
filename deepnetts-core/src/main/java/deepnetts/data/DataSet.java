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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A collection of data set items.
 * 
 * thos should be the interface in visrec.ml
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 * @param <ITEM_TYPE>
 */
public class DataSet<ITEM_TYPE extends DataSetItem> implements Iterable<ITEM_TYPE> {
    List<ITEM_TYPE> items;
    private Iterator<ITEM_TYPE> iterator;

    private String label;
     
    // constructor with vector dimensions annd capacity
     
    public DataSet() {
        items = new ArrayList<>();
    }
               
    @Override
    public Iterator<ITEM_TYPE> iterator() {
        iterator =  items.iterator();
        return iterator;
    }

    public void add(ITEM_TYPE item) {
        items.add(item);
    }

    public int size() {
        return items.size();
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
            
}
