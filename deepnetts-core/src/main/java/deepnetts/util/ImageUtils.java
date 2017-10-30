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

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class ImageUtils {
        
    /**
     * Scales input image to specified target width or height, centers and returns resulting image.
     * Scaling factor is calculated using larger dimension (width or height).
     * Keeps aspect ratio and image type, and bgColor parameter to fill background. 
     * 
     * @param img
     * @param targetWidth
     * @param targetHeight
     * @param bgColor
     * @return scaled and centered image
     */
    public static BufferedImage scaleAndCenter(BufferedImage img, int targetWidth, int targetHeight, int padding, Color bgColor) {

        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
                
        float scale = 0;
        int xPos, yPos;
        
        if (imgWidth > imgHeight) {
            scale = imgWidth / (float)(targetWidth-2*padding);

        } else { // imgHeight < imgWidth
            scale = imgHeight / (float)(targetHeight-2*padding);

        }

        int newWidth = (int) (imgWidth / scale);
        int newHeight = (int)(imgHeight / scale);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
        
        BufferedImage resultImg = new BufferedImage(targetWidth, targetHeight, img.getType());
        resultImg.getGraphics().setColor(bgColor);
        resultImg.getGraphics().fillRect(0, 0, targetWidth, targetHeight);
                
        if (imgWidth > imgHeight) {
            xPos = padding;
            yPos = padding + (targetHeight-2*padding - newHeight) / 2;            
        } else {
            xPos = padding + (targetWidth -2*padding - newWidth) / 2;
            yPos = padding;                                    
        }
        
        resultImg.getGraphics().drawImage(scaledImg, xPos, yPos, null);
                    
        return resultImg;
    }

    /**
     * Scales input image to specified target width or height, crops and returns resulting image.
     * Scaling factor is calculated using smaller dimension (width or height).
     * Keeps aspect ratio and image type, and bgColor parameter to fill background. 
     * 
     * @param img image to scale
     * @param targetWidth target image width
     * @param targetHeight target image height
     * @return scaled and cropped image
     */
    public static BufferedImage scaleBySmallerAndCrop(BufferedImage img, int targetWidth, int targetHeight) {

        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
                
        float scale = 0;

        
        if (imgWidth < imgHeight) { // which one is smaller width or height?
            scale = imgWidth / (float)targetWidth; // scale by width
        } else { // imgHeight < imgWidth // scale by height
            scale = imgHeight / (float)targetHeight;
        }

        int newWidth = (int) (imgWidth / scale);
        int newHeight = (int)(imgHeight / scale);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
               
        BufferedImage scaledBuffImg = new BufferedImage(newWidth, newHeight, img.getType());
        scaledBuffImg.getGraphics().drawImage(scaledImg, 0, 0, null);
        
        BufferedImage resultImage = null;
        
        if (imgWidth < imgHeight) { // crop by centering on  height
            final int xPos = 0;
            final int yPos = (newHeight - targetHeight) / 2;            
            resultImage = scaledBuffImg.getSubimage(xPos, yPos, targetWidth, targetHeight);            
        } else { // crop by centering on height
            final int xPos = (newWidth-targetWidth) / 2;
            final int yPos = 0;                                    
            resultImage = scaledBuffImg.getSubimage(xPos, yPos, targetWidth, targetHeight);            
        }
        
        return resultImage;
    }
    
    
    /**
     * Loads all images from the specified directory, and returns them as a list.
     * 
     * @param dir
     * @return list of images as BufferedImage instances
     * @throws IOException 
     */
    public static List<BufferedImage> loadImagesFromDirectory(File dir) throws IOException {
        List<BufferedImage> imageList = new ArrayList<>();
        for (final File file : dir.listFiles()) {
            if (!file.isDirectory()) {
                BufferedImage img = ImageIO.read(file);
                imageList.add(img);
            } 
        }        
        return imageList;
    }
    
    
    /**
     * Loads images (jpg, jpeg, png) from specificed directory and returns them as a map with File object as a key and BufferedImage object as a value.
     *  
     * @param dir
     * @return
     * @throws IOException 
     */
    public static HashMap<File, BufferedImage> loadFileImageMapFromDirectory(File dir) throws IOException {
        if (!dir.isDirectory()) throw new IllegalArgumentException("Parameter dir must be a directory: "+dir.toString());
        
        HashMap<File, BufferedImage> imageMap = new HashMap<>();
        for (final File file : dir.listFiles()) {
            if (file.isDirectory()) continue;// skip subdirectories
            
            final String imgType = getImageType(file);                       
            if (!imgType.equalsIgnoreCase("jpg") && !imgType.equalsIgnoreCase("jpeg") && !imgType.equalsIgnoreCase("png")) continue;
                                        
            BufferedImage img = ImageIO.read(file);
            imageMap.put(file, img);            
        }        
        return imageMap;
    }    
    
    public static String getImageType(final File file) {
            final String fileName = file.getName();
            return fileName.substring(fileName.lastIndexOf(".")+1); // get file ext/img type                       
    }
    
    
    // hardcoded for lego replace with DataSetUtils
    public static void createIndexFile(HashMap<File, BufferedImage> imageMap, String imageFile, boolean useAbsPath) throws IOException {
        // dodaj regex pomocu koga ce da ih ubacuje uklas?
        try (BufferedWriter out = new BufferedWriter(new FileWriter(imageFile))) {
            int fileCount = imageMap.size();
            int i = 0;

            for (File file : imageMap.keySet()) {
                if (!useAbsPath) 
                    out.write(file.getName() + " legoman"); // TODO: how to get image label?
                else
                    out.write(file.getPath() + " legoman");
                
                if (i < fileCount - 1) {
                    out.write(System.lineSeparator());
                }
                i++;
            }
        }
    }
    
    
    /**
     *  Returns an array of images created by translating specified input image 
     *  by specified number of count with specified step size.
     * 
     * @param img image to translate
     * @param stepCount number of 
     * @param stepSize 
     * @return 
     */
    public static BufferedImage[] randomTranslateImage(BufferedImage img, int stepCount, int stepSize) {
        // u krug za dati radijus
        // random distance
        BufferedImage[] images = new BufferedImage[stepCount*4];  
        AffineTransform[] trans = new AffineTransform[4];
        
        for(int i = 0; i < stepCount; i++) {
            // pomeri na sve cetiri strane po count * step pixela
            trans[0] = AffineTransform.getTranslateInstance(i*stepSize, 0);
            trans[1] = AffineTransform.getTranslateInstance(-i*stepSize, 0);
            trans[2] = AffineTransform.getTranslateInstance(0, i*stepSize);
            trans[3] = AffineTransform.getTranslateInstance(0, -i*stepSize);            
            
            for (int t = 0; t < 4; t++) {
                BufferedImage newImage = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
                Graphics2D dispGc = newImage.createGraphics();
                dispGc.setBackground(Color.WHITE);
                dispGc.clearRect(0, 0, newImage.getWidth(), newImage.getHeight());
                dispGc.drawImage(img, trans[t], null);
                images[i*4+t] = newImage;
            }
        }
        
        return images;        
    }
    

    /**
     * Crops specified number of random subimages of specified dimensions.
     * 
     * @param img   main image to crop
     * @param cropWidth width of the cropped image
     * @param cropHeight height of the croped image
     * @param cropNum number of images to crop
     * @param random random number generator used to generate random positions
     * @return list of randomly cropped images
     */
    public static List<BufferedImage> randomCrop(BufferedImage img, int cropWidth, int cropHeight, int cropNum, Random random) {
        List<BufferedImage> croppedImages = new ArrayList<>();
        
        final int maxX = img.getWidth() - cropWidth;
        final int maxY = img.getHeight() - cropHeight;
        
        for(int i=0; i< cropNum; i++) {
            final int x = random.nextInt(maxX);
            final int y = random.nextInt(maxY);
            BufferedImage cropped = img.getSubimage(x, y, cropWidth, cropHeight);
            croppedImages.add(cropped);
        }
        
        return croppedImages;
    }
    
    /**
     * Still not working as it should
     * @param img
     * @param tint
     * @param brightness
     * @return 
     * 
     * https://en.wikipedia.org/wiki/Normalization_(image_processing)
     * 
     */
    public static List<BufferedImage> randomTintAndBrightness(BufferedImage img, float maxTint, int maxBrightness, int num, Random random) {
        // hinton mnozi sopstene vektore i dodaje ih na asliku ima objasnjeno i umagenet radu
        List<BufferedImage> augmentedImages = new ArrayList<>();
        // nekako klimavo ali nesto radi samo sa dodavanjem 
        for(int i=0; i< num; i++)  {
            final float k = random.nextFloat()*maxTint;
            final int n = random.nextInt(maxBrightness);
            for (int y = 0; y < img.getHeight(); y++) {
                for (int x = 0; x < img.getWidth(); x++) {
                    int rgb = img.getRGB(x, y);
                    //rgb = (int)(  k*rgb);
                    rgb = rgb+n;
                    if (rgb>255) rgb=255;
                    img.setRGB(x, y, rgb);
                }
            }       
            BufferedImage newImg= deepCopy(img);
            augmentedImages.add(newImg);            
        }
        
        return augmentedImages;
    }
    
    
    static BufferedImage deepCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(null);
        // https://stackoverflow.com/questions/3514158/how-do-you-clone-a-bufferedimage
      //  return new BufferedImage(cm, raster, isAlphaPremultiplied, null).getSubimage(0, 0, bi.getWidth(), bi.getHeight());
         return new BufferedImage(cm, raster, isAlphaPremultiplied, null); 
    }    
    
    public static void writeImages(List<BufferedImage> images, String targetPath, String fileName, String fileType) throws IOException {
        int i=0;
        for(BufferedImage img : images) {
            i++;
            ImageIO.write(img, fileType, new File(targetPath + File.separator  +fileName+"_"+i+"."+fileType));
        }
    }
    
    
    /**
     * Generates a specified number of randomly full colored images of specified size.
     * Images are saved at specified destination path named using negative_x.jpg name pattern
     * 
     * @param width image width
     * @param height image height
     * @param numberOfImages number of images to generate
     * @param destPath destination path to save images
     * 
     * @throws IOException 
     */
    public static void generateRandomColoredImages(int width, int height, int numberOfImages, String destPath) throws IOException{
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        
        Graphics gr = image.getGraphics();
        float r, g, b;
        
        for (int i = 0; i < numberOfImages; i++) {
            r = (float) Math.random();
            g = (float) Math.random();
            b = (float) Math.random();

            gr.setColor(new Color(r, g, b));
            gr.fillRect(0, 0, width, height);
            String fileName = "negative_"+i+".jpg";

            ImageIO.write(image, "jpg", new File(destPath + "/" + fileName));
        }        
    }
    
    public static void generateNoisyImage(int width, int height, int numberOfImages, String destPath) throws IOException {
        //create buffered image object img
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        //file object
        for (int i = 0; i < numberOfImages; i++) {
            //create random image pixel by pixel
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int a = (int) (Math.random() * 256); //alpha
                    int r = (int) (Math.random() * 256); //red
                    int g = (int) (Math.random() * 256); //green
                    int b = (int) (Math.random() * 256); //blue

                    int color = (a << 24) | (r << 16) | (g << 8) | b; //pixel

                    image.setRGB(x, y, color);
                }
            }
            String fileName = "noisy_" + i + ".jpg";

            ImageIO.write(image, "jpg", new File(destPath + "/" + fileName));
        }

    }    
    
   
    
    
}
