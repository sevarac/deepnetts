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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
    
package deepnetts.data;

import deepnetts.core.DeepNetts;
import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Represents data set with images
 * 
 * @author zoran
 */
public class ImageSet extends DataSet<ExampleImage> { 
    // ovi ne mogu svi da budu u memoriji odjednom...
    // this should be items
    private final List<ExampleImage> images; // mozda neka konkurentna kolekcija da vise threadova moze paralelno da trenira i testira nekoliko neuronskih mreza
    private final List<String> labels;
    private final int imageWidth;
    private final int imageHeight;
    private Tensor mean;
    
    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());    
        
   // osmisliti i neki protocni / buffered data set, koji ucitava jedan batch
      
    public ImageSet(int imageWidth, int imageHeight) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        
        images = new ArrayList();       
        labels = new ArrayList();     
    }    

    public ImageSet(int imageWidth, int imageHeight, int capacity) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        
        images = new ArrayList(capacity);       
        labels = new ArrayList();     
    }        
    
    /**
     * Adds image to this image set.
     * 
     * @param image
     * @throws DeepNettsException if image is empty or has wrong dimensions.
     */
    public void add(ExampleImage image) throws DeepNettsException {
        if (image == null) throw new DeepNettsException("Example image cannot be null!");
        
        if ((image.getWidth() == imageWidth) && (image.getHeight() == imageHeight)) {        
            images.add(image);
        } else {
            throw new DeepNettsException("Wrong image dimensions for this data set. All images should be "+imageWidth + "x"  + imageHeight);
        }        
    }
    
    
    /**
     * Loads example images with labels from specified file.
     * 
     * TODO: First load entire image index, then load and preprocess image in multihredaded way
     * 
     * @param imageIdxFile Plain text file that contains space delimited image file paths and labels
     * @throws java.io.FileNotFoundException if imageIdxFile was not found
     */    
    public void loadImages(File imageIdxFile, boolean absPaths) throws FileNotFoundException  {
        String parentPath = "";
        if (absPaths == false) {
            parentPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));
        }
        
        String imgFileName=null;
        String label=null;
        
        try(BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;        
            // we can also catch and log FileNotFoundException, IOException in this loop
            while((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] str = line.split(" "); // parse file and class label from current line - sta ako naziv fajla sadrzi space? - to ne sme ili detektuj nekako sa lastIndex
                
                imgFileName = str[0];
                if (!absPaths) imgFileName = parentPath + File.separator + imgFileName;
                if (str.length == 2) {
                    label = str[1];
                } else if (str.length ==1) {
                    // todo: extract label from parent folder
                }
                
                // todo: load and preprocess image pixels in separate thread in batches of 10 images
                ExampleImage exImg = new ExampleImage(new File(imgFileName), label);
                exImg.setTargetOutput(createOutputVectorFor(label));
                
                // make sure all images are the same size
                if ((exImg.getWidth() != imageWidth) || (exImg.getHeight() != imageHeight)) throw new DeepNettsException("Bad image size for "+exImg.getFile().getName());    
                
                images.add(exImg);  
            }
            
            if (images.isEmpty()) throw new DeepNettsException("Zero images loaded!");           
            
            LOGGER.info("Loaded "+images.size()+ " images");
            
        } catch (FileNotFoundException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }
 
    }
    
    
    /**
     * Loads example images and corresponding labels from specified file.
     * 
     * @param imageIdxFile Plain text file which contains space delimited image file paths and labels
     * @param numOfImages number of images to load
     */    
    public void loadImages(File imageIdxFile, boolean absPaths, int numOfImages) throws DeepNettsException  {
        String parentPath = "";
        if (absPaths == false) {
            parentPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));
        }
        
        String imgFileName=null;
        String label=null;
        // ako je numOfImages manji od broja slika u fajlu logovati
          try (BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {

            String line = null;

            for (int i = 0; i < numOfImages; i++) {
                line = br.readLine();
                if (line.isEmpty()) {
                    continue;
                }
                String[] str = line.split(" "); // parse file and class label from line
                imgFileName = str[0];
                if (!absPaths) imgFileName = parentPath + File.separator + imgFileName;
                label = str[1];
                
                ExampleImage exImg = new ExampleImage(new File(imgFileName), label);
                exImg.setTargetOutput(createOutputVectorFor(label));

                // make sure all images are the same size
                if ((exImg.getWidth() != imageWidth) || (exImg.getHeight() != imageHeight)) {
                    throw new DeepNettsException("Bad image size!");
                }

                images.add(exImg);
            }

            if (images.isEmpty()) {
                throw new DeepNettsException("Zero images loaded!");
            }
            LOGGER.info("Loaded " + images.size() + " images");

        } catch (FileNotFoundException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }
    }    

    /**
     * Creates and returns binary target vector for specified label using one-of-many scheme.
     * Returns all zeros for label 'negative'.
     * 
     * @param label
     * @return 
     */
    private float[] createOutputVectorFor(final String label) {
        final float[] out = new float[labels.size()];
        
        if (label.equals("negative")) return out;
        
        final int idx = labels.indexOf(label); // get label index
        if (idx != -1) {
            out[idx] = 1;
            return out;
        }   
            
        throw new DeepNettsException("Label '"+label+"' not found in labels file!");
    }
    
    
    
    public List<ExampleImage> getImages() {
        return images;
    }

    public List<String> getOutputLabels() {
        return labels;
    }
    
    public int getLabelsCount() {
        return labels.size();
    }
    
    // shuffle should have random seed?
    public void shuffle() {
        Random rnd = RandomGenerator.getDefault().getRandom();
        Collections.shuffle(images, rnd); // use one with rand param
    }
    
    public void shuffle(int seed) {
        Random rnd = new Random(seed);
        Collections.shuffle(images, rnd);
    }    
    
    
    public ImageSet[] split(int ... percents) {               
        int pSum=0;
        for(int i=0; i<percents.length; i++)
            pSum += percents[i];
        
        if (pSum > 100) throw new DeepNettsException("Sum of percents cann not be larger than 100!");
        
        int idx=0;
        ImageSet[] subsets = new ImageSet[percents.length];
        shuffle();       
        for(int i=0; i<percents.length; i++) {
             ImageSet imageSubSet = new ImageSet(imageWidth, imageHeight); 
             int itemsCount =(int) (size() * percents[i] / 100.0f);
             
             for(int j=0; j<itemsCount; j++) {
                 imageSubSet.add(images.get(idx));
                 idx++;
             }
             
             subsets[i] = imageSubSet;
             subsets[i].labels.addAll(labels);
        }
                        
        return subsets;
    }
        
     public List<String> loadLabels(String filePath) throws DeepNettsException {
         return loadLabels(new File(filePath));
     }
    
    public List<String> loadLabels(File file) throws DeepNettsException {
        try(BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = null;        
            // we can also catch and log FileNotFoundException, IOException in this loop
            while((line = br.readLine()) != null) {                              
                labels.add(line);               
            }
            br.close();
            return labels;
        } catch (FileNotFoundException ex) {            
              LOGGER.error("Could not find labels file: "+file.getAbsolutePath(), ex);
              throw new DeepNettsException("Could not find labels file: "+file.getAbsolutePath(), ex);
        } catch (IOException ex) {
            LOGGER.error("Error reading labels file: "+file.getAbsolutePath(), ex);
            throw new DeepNettsException("Error reading labels file: "+file.getAbsolutePath(), ex);
        }        
    }

    @Override
    public int size() {
       return images.size();
    }
    
    /**
     * Applies zero mean normalization to entire dataset, and returns mean matrix.
     * 
     * @return Returns mean matrix for the entire dataset
     */
    public Tensor zeroMean() {
        mean = new Tensor(imageHeight, imageWidth, 3);
        
        // sum all matrices
        for(ExampleImage image : images) {
            mean.add(image.getInput());
        }
        
        // divide by number of images
        mean.div((float)images.size());
        
        // subtract mean from each image
        for(ExampleImage image : images) {
            image.getInput().sub(mean);
        }        
        
        return mean;
    }
    
    public void invert() {
        for(ExampleImage image : images) {
        //    mean.add(image.getInput());
            float[] rgbVector = image.getRgbVector();
            for(int i=0; i<rgbVector.length; i++) {
                rgbVector[i] =  1-rgbVector[i];
            }
        }        
    }
    
    
    
    @Override
    public Iterator<ExampleImage> iterator() {        
        return images.iterator();
    }    
    

}
