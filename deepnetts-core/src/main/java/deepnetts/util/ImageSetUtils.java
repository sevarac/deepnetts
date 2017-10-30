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

import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class ImageSetUtils {

    public static final String LABELS_FILE = "labels.txt";
    public static final String TRAIN_FILE = "train.txt";
    public static final String TEST_FILE = "test.txt";    
    public static final String IMAGE_IDX_FILE = "index.txt";    
    
    private static final Logger LOGGER = Logger.getLogger(ImageSetUtils.class.getName());
    
    
  /**
   * Creates image data set from raw images by resizing and randomly croping to target dimensions.
   * Scale image by smaller dimension to target width/height
   * Crop image on center to fit target dimensions
   * Save all images to target path
   *
   * This how its done for AlexNet/ImageNet competition
   * 
   * @param sourcePath
   * @param targetPath
   * @param targetWidth
   * @param targetHeight
   * @param useAbsolutePaths
   * @return
   * @throws IOException 
   */
  public static ImageSet createDataSetFromRawImages(String sourcePath, String targetPath, int targetWidth, int targetHeight, boolean useAbsolutePaths) throws IOException {
    ImageSet imageSet = new ImageSet(targetWidth, targetHeight);
      
    // list all subddirectories / categories
    List<String> labels = labelsFromSubDirectories(sourcePath);
    List<String> imgIndex = new ArrayList<>();
        
    for(final String label: labels) { // for all subdirectories (parallel stream)
        createdDirectory(new File(targetPath + File.separator + label));
        // load all images from target path into memory
        Map<File, BufferedImage> categoryImages = ImageUtils.loadFileImageMapFromDirectory((new File(sourcePath + File.separator + label))); // make sure we're loading only image files
        int count = 0;
        for(final Map.Entry<File, BufferedImage> entry : categoryImages.entrySet()) {        
            // scale to smaller dimension and crop imagecenter

            final String fileName = entry.getKey().getName();

            LOGGER.info("Processing image " + fileName);
            count++;
            
            // scale image an add it to image set
            final BufferedImage scaledImage = ImageUtils.scaleBySmallerAndCrop(entry.getValue(), targetWidth, targetHeight);
            imageSet.add(new ExampleImage(scaledImage, label));
            
            // write scaled image to target dir                       
            final String imgType = ImageUtils.getImageType(entry.getKey());            
            final String targetFile = targetPath + File.separator + label + File.separator + label+"_"+count+"."+imgType; // make sure the key is only filename without the path
            ImageIO.write(scaledImage, imgType, new File(targetFile)); // ovde pisi i png fajlocve aimenaim numerisi sa 123
            
            // and put it in image index (add labels to generated files too)
            if (useAbsolutePaths) {
                imgIndex.add(targetFile + " " + label);
            } else {
                imgIndex.add(label + File.separator + fileName + " " + label);
            }
            
            // create augmented images here?
            // 
            
            LOGGER.info(targetFile + " done!");
        }                
    }
                    
    // create labels.txt
    writeToFile(labels, targetPath + File.separator +LABELS_FILE);
    
    // create index.txt
    writeToFile(imgIndex, targetPath + File.separator + IMAGE_IDX_FILE);
    
    return imageSet;
  }
  
  // ovo i prethodnu metodi najbolje na kraju staviti u neku klasu koja sve to radi, ili napraviti onaj pipeline
  public static void createRandomlyCroppedImages(String srcPath, String destPath, int targetWidth, int targetHeight, int num) throws IOException {
   List<String> labels = labelsFromSubDirectories(srcPath);
    List<String> imgIndex = new ArrayList<>();
    Random random = new Random(123);
        
    for(final String label: labels) { // for all subdirectories (parallel stream)
        createdDirectory(new File(destPath + File.separator + label));
        // load all images from target path into memory
        Map<File, BufferedImage> categoryImages = ImageUtils.loadFileImageMapFromDirectory((new File(srcPath + File.separator + label))); // make sure we're loading only image files
        int count = 0;
        for(final Map.Entry<File, BufferedImage> entry : categoryImages.entrySet()) {        
            // scale to smaller dimension and crop imagecenter

            final String fileName = entry.getKey().getName();

            LOGGER.info("Processing image " + fileName);
            count++;
            
            // scale image an add it to image set
            List<BufferedImage> randomlyCropedImages = ImageUtils.randomCrop(entry.getValue(), targetWidth, targetHeight, num, random); 
           // imageSet.add(new ExampleImage(scaledImage, label));
            
            // write scaled image to target dir                       
            final String imgType = ImageUtils.getImageType(entry.getKey());            
            final String targetFile =  label + File.separator + label+"_"+count; // make sure the key is only filename without the path

            ImageUtils.writeImages(randomlyCropedImages, destPath, targetFile, imgType);
            
            
//ImageIO.write(scaledImage, imgType, new File(targetFile)); // ovde pisi i png fajlocve aimenaim numerisi sa 123
            
            // and put it in image index (add labels to generated files too)
//            if (useAbsolutePaths) {
//                imgIndex.add(targetFile + " " + label);
//            } else {
//                imgIndex.add(label + File.separator + fileName + " " + label);
//            }
            
            // create augmented images here?
            // 
            
            LOGGER.info(targetFile + " done!");
        } 
        // new index cn be created here with separate method
    }      
  }  
  
    private static void  createdDirectory(File dir) {
       if (!dir.exists()) dir.mkdir();
    }
    
  /**
   * Returns a list of category/class labels from the names of subdirectories for the given path.
   * Use this method to generate labels for example images.
   * Expected directory structure is to have one subdirecotry for each category/class.
   * 
   * @param path path to search for subirectories
   * @return list of class labels
   */  
  public static List<String> labelsFromSubDirectories(String path) {
        List<String> labels = new ArrayList<>();        
        File rootDir = new File(path);
        if (!rootDir.isDirectory()) throw new DeepNettsException("Specified path must be a directory: "+path);        
        String[] subDirs = rootDir.list();

        for(String dir : subDirs){
            if (new File(path+"/"+dir).isDirectory()) {
                labels.add(dir);
            }
        }
        
        return labels;
    }
    
   /**
    * Writes a specified list of strings to file.
    * 
    * @param list a list of strings to write to file
    * @param fileName name of the file to write to
    * @throws java.io.IOException
    */
    public static void writeToFile(List<String> list, String fileName) throws IOException {
        File file = new File(fileName);
        
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file)))) {
            list.forEach((item) -> {
                pw.println(item);
            });
        }
    }
    
    /**
     * Creates a labels index file from subdirectories at the given path.
     * Labels in file are sorted by name. Directory names should correspond to category/class labels.
     * 
     * @param path path to directory which contains image categories/classes
     * @throws IOException 
     */    
    public static void createLabelsIndex(String path) throws IOException {
        List<String> labels = labelsFromSubDirectories(path);
        labels.remove("negative"); // dont put negative label if its there
        Collections.sort(labels);
        writeToFile(labels, path + File.separator + LABELS_FILE);        
    }
    
   /**
    * List all files in all subdirectories and write them into single index.txt file.
    * index.txt is created in the same directory specifed as path
    * Expected structure of directory is /imageCategoryName/imageFile1.jpg - each category of images should have its own directory
    * which is named as corresponding category
    * 
    * TODO: sta se desava ako index.txt vec postoji? Da upozorava?
    * 
    * @param path path to indes
    * @param useFullPath true if image index should contain full image path, false otherwise
    * @throws java.io.IOException 
    */    
    public static File createImageIndex(String path, boolean useFullPath) throws IOException { // provide a path to train or text dir
     
        File rootDir = new File(path);
        if (!rootDir.isDirectory()) throw new DeepNettsException("Specified path must be a directory: "+path);            

        File imageIdxFile = new File(path +"/" +IMAGE_IDX_FILE);      
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(imageIdxFile)));
        
        String[] subDirs = rootDir.list();

        for(String classDirName : subDirs){ // these are the class directories            
            File classDirFile = new File(path+"/"+classDirName);
             if (!classDirFile.isDirectory()) continue;
            String label = classDirName;
            // get label index from labels file
         
           String[] imageFiles = classDirFile.list();
           for(String imageFile : imageFiles) {
               if (useFullPath)
                    pw.println(path + "/" + classDirName + "/"+imageFile + " " + label);                        
               else 
                   pw.println(classDirName + "/" + imageFile + " " + label);                        
           }
        }   
        
        pw.close();
      
        return imageIdxFile;
    }
                
    /**
     * Renames files in  specified directory. 
     * New file names correspond to numerated image category/class label.
     * className_0001.jpg, className_002.jpg, ...
     * 
     * @param srcPath
     * @param className
     * @throws FileNotFoundException 
     */
    public static void renameFilesAsClasses(String srcPath, String className) throws FileNotFoundException {
        File sourceDir = new File(srcPath);
        
        if (!sourceDir.exists()) throw new FileNotFoundException("Source directory '"+srcPath+"' not found!");
        if (!sourceDir.isDirectory()) throw new DeepNettsException(srcPath + "is not a directory!");        
               
        File files[] = sourceDir.listFiles();
        
        int z = 1;

        while((Math.pow(10, z) < files.length)) { 
            z++;
        }
                
        
        for(int i=0; i<files.length; i++) {
            String leadingZeros=getLeadingZeros(i, z);
            
            if (files[i].renameTo(new File(srcPath+"/"+className+"_"+leadingZeros + i+ ".jpg"))) {   // a sta ako je png - uzmi u obzir tip fajla
                System.out.println("Renamed "+className+"_"+i);
            }
        }
    }
            
    /**
     * Copies specified number of samples of each class from 
     * @param srcDir
     * @param targetFileName
     * @param samplesPerClass
     * @param useFullPath
     * @return
     * @throws IOException 
     */    
    public static File createSubSampledImageIndex(String srcDir, String targetFileName, int samplesPerClass, boolean useFullPath) throws IOException { // provide a path to train or text dir
      // list all files in all subdirectories and write them into single file  
        File imageIdxFile = new File(targetFileName);
      
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(imageIdxFile)))) {
            File file = new File(srcDir);
            String[] subDirs = file.list();
            
            for(String classDirName : subDirs){ // iterate class
                File classDirFile = new File(srcDir+"/"+classDirName);
                if (!classDirFile.isDirectory()) continue;
                
                String label = classDirName;
                // get label index from labels file
                
                String[] imageFiles = classDirFile.list();
                for(int i=0; i<samplesPerClass; i++) {  // samplesPerClass
                    String imageFile = imageFiles[i];
                    
                    if (useFullPath)
                        pw.println(srcDir + "/" + classDirName + "/"+imageFile + " " + label);
                    else
                        pw.println(classDirName + "/" + imageFile + " " + label);
                }
            }
        }
      
        return imageIdxFile;
    }    


    
    
    private static String getLeadingZeros(int i, int z) {
        String leadingZeros = "";
        int digits = (""+i).length();
        
        int a=0;
        while(a < (z-digits)) {
            leadingZeros += "0";
            a++;
        }
                                   
        return leadingZeros;
    }
    
    
    public static void main(String[] args) {
        // get opetions from args  and invoke corresponding mehods
    }
    
}
