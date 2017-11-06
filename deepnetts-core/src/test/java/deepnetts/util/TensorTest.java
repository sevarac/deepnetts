package deepnetts.util;

import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class TensorTest {
      
    public TensorTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
        // create default tensor to test here
        
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void testGet_int() {
        int idx = 2;
        Tensor instance = new Tensor(new float[] {0.1f, 0.2f, 0.3f, 0.4f});
        float expResult = 0.3F;
        float result = instance.get(idx);
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testSet_int_float() {
        int idx = 2;
        float val = 0.3F;
        Tensor instance = new Tensor(5);
        float result = instance.set(idx, val);        
        float expResult = 0.3F;        
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testGet_int_int() {
        int row = 1;
        int col = 2;
        Tensor instance = new Tensor(new float[][] {{0.11f, 0.12f, 0.13f}, {0.21f, 0.22f, 0.23f}});
        float expResult = 0.23f;
        float result = instance.get(row, col);
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testSet_3args() {
        int row = 1;
        int col = 1;
        float val = 0.37F;
        Tensor instance = new Tensor(new float[][] {{0.11f, 0.12f, 0.13f}, {0.21f, 0.22f, 0.23f}});
        instance.set(row, col, val);
        float expResult = val;
        float result = instance.get(row, col);        
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testGet_3args() {
        int row = 1;
        int col = 2;
        int z = 1;
                                          //  z,r,c          
        Tensor instance = new Tensor(new float[][][] { {{0.111f, 0.121f, 0.131f, 0.141f}, 
                                                        {0.211f, 0.221f, 0.231f, 0.241f},
                                                        {0.311f, 0.321f, 0.331f, 0.341f}}, 
            
                                                        {{0.112f, 0.122f, 0.132f, 0.142f},
                                                         {0.212f, 0.222f, 0.232f, 0.242f},
                                                         {0.312f, 0.322f, 0.332f, 0.342f}}});
        float expResult = 0.232f;
        float result = instance.get(row, col, z);                        
        
//        row=1; col=0; z=1;
//        float expResult = 0.212F;
//        float result = instance.get(row, col, z);                        
        assertEquals(expResult, result, 0.0);        
        
    }

    @Test
    public void testSet_4args() {
                                          //  z,r,c          
        Tensor instance = new Tensor(new float[][][] { {{0.111f, 0.121f, 0.131f, 0.141f}, 
                                                        {0.211f, 0.221f, 0.231f, 0.241f},
                                                        {0.311f, 0.321f, 0.331f, 0.341f}}, 
            
                                                        {{0.112f, 0.122f, 0.132f, 0.142f},
                                                         {0.212f, 0.222f, 0.232f, 0.242f},
                                                         {0.312f, 0.322f, 0.332f, 0.342f}}});
        int row = 2;
        int col = 3;
        int z = 1;
        float val = 0.777F;

        instance.set(row, col, z, val);
        float expResult = val;        
        float result = instance.get(row, col, z);         
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testGet_4args() {
                                           // f, z,r,c          
        Tensor instance = new Tensor(new float[][][][] { {{{0.1111f, 0.1121f, 0.1131f, 0.1141f}, 
                                                          {0.1211f, 0.1221f, 0.1231f, 0.1241f},
                                                          {0.1311f, 0.1321f, 0.1331f, 0.1341f}}, 
            
                                                         {{0.1112f, 0.1122f, 0.1132f, 0.1142f},
                                                          {0.1212f, 0.1222f, 0.1232f, 0.1242f},
                                                          {0.1312f, 0.1322f, 0.1332f, 0.1342f}}},
                                                         
                                                         {{{0.2111f, 0.2121f, 0.2131f, 0.2141f}, 
                                                          {0.2211f, 0.2221f, 0.2231f, 0.2241f},
                                                          {0.2311f, 0.2321f, 0.2331f, 0.2341f}}, 
            
                                                         {{0.2112f, 0.2122f, 0.2132f, 0.2142f},
                                                          {0.2212f, 0.2222f, 0.2232f, 0.2242f},
                                                          {0.2312f, 0.2322f, 0.2332f, 0.2342f}}}        
                                                        });
        
        int row = 2;
        int col = 3;
        int z = 1;
        int fourth = 1;

        float expResult = 0.2342f;
        float result = instance.get(row, col, z, fourth);
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testSet_5args() {
                                           // f, z,r,c          
        Tensor instance = new Tensor(new float[][][][] { {{{0.1111f, 0.1121f, 0.1131f, 0.1141f}, 
                                                          {0.1211f, 0.1221f, 0.1231f, 0.1241f},
                                                          {0.1311f, 0.1321f, 0.1331f, 0.1341f}}, 
            
                                                         {{0.1112f, 0.1122f, 0.1132f, 0.1142f},
                                                          {0.1212f, 0.1222f, 0.1232f, 0.1242f},
                                                          {0.1312f, 0.1322f, 0.1332f, 0.1342f}}},
                                                         
                                                         {{{0.2111f, 0.2121f, 0.2131f, 0.2141f}, 
                                                          {0.2211f, 0.2221f, 0.2231f, 0.2241f},
                                                          {0.2311f, 0.2321f, 0.2331f, 0.2341f}}, 
            
                                                         {{0.2112f, 0.2122f, 0.2132f, 0.2142f},
                                                          {0.2212f, 0.2222f, 0.2232f, 0.2242f},
                                                          {0.2312f, 0.2322f, 0.2332f, 0.2342f}}}        
                                                        });
        
        int row = 2;
        int col = 3;
        int z = 1;
        int f=1;
        float val = 0.999f;

        instance.set(row, col, z, f, val);
        float expResult = val;     // 2342f           
        float result = instance.get(row, col, z, f);         
        assertEquals(expResult, result, 0.0);        
    }

    @Test
    public void testGetValues() {
        float[] values = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        Tensor instance = new Tensor(values);
        float[] result = instance.getValues();
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};        
        assertArrayEquals(expResult, result, 0.0f);
    }

    @Test
    public void testSetValues() {
        float[] values = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        Tensor instance = new Tensor(5);
        instance.setValues(values);
        float[] result = instance.getValues();
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        assertArrayEquals(expResult, result, 0.0f);
    }

    @Test
    public void testGetCols() {
        Tensor instance = new Tensor(5);        
        int expResult = 5;
        int result = instance.getCols();
        assertEquals(expResult, result);
    }

    @Test
    public void testGetRows() {
        int rows = 5;
        int cols = 4;        
        Tensor instance = new Tensor(5, 4);        
        int expResult = 5;
        int result = instance.getRows();
        assertEquals(expResult, result);
    }

    @Test
    public void testGetDepth() {
        int rows = 3;
        int cols = 4;
        int depth = 5;
        Tensor instance = new Tensor(rows, cols, depth);
        int expResult = depth;
        int result = instance.getDepth();
        assertEquals(expResult, result);
    }

    @Test
    public void testGetFourthDim() {
        int rows = 3;
        int cols = 4;
        int depth = 5;
        int fourth = 6;
        Tensor instance = new Tensor(rows, cols, depth, fourth);
        int expResult = fourth;
        int result = instance.getFourthDim();
        assertEquals(expResult, result);
    }

    @Test
    public void testGetDimensions() {
        int rows = 3;
        int cols = 4;
        int depth = 5;
        Tensor instance = new Tensor(rows, cols, depth);
        int expResult = 3;
        int result = instance.getDimensions();
        assertEquals(expResult, result);
    }

    @Test
    public void testToString() {
        float[] values = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        Tensor instance = new Tensor(values);
        String expResult = "[0.1, 0.2, 0.3, 0.4, 0.5]";
        String result = instance.toString();
        assertEquals(expResult, result);
    }

    @Test
    public void testAdd_3args() {
        Tensor instance = new Tensor(new float[][] {{0.11f, 0.12f, 0.13f}, {0.21f, 0.22f, 0.23f}});
        int row = 1;
        int col = 2;
        float value = 0.1f;        
        
        instance.add(row, col, value);                
        float expResult = 0.33f;                
        assertEquals(expResult, instance.get(row, col), 0.0);
    }

    @Test
    public void testAdd_4args() {
                                          //  z,r,c          
        Tensor instance = new Tensor(new float[][][] { {{0.111f, 0.121f, 0.131f, 0.141f}, 
                                                        {0.211f, 0.221f, 0.231f, 0.241f},
                                                        {0.311f, 0.321f, 0.331f, 0.341f}}, 
            
                                                        {{0.112f, 0.122f, 0.132f, 0.142f},
                                                         {0.212f, 0.222f, 0.232f, 0.242f},
                                                         {0.312f, 0.322f, 0.332f, 0.342f}}});
        int row = 2;
        int col = 3;
        int z = 1;
        float val = 0.12f;

        instance.add(row, col, z, val);
        float expResult = 0.462f;                
        float result = instance.get(row, col, z);         
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testAdd_5args() {
        Tensor instance = new Tensor(new float[][][][] { {{{0.1111f, 0.1121f, 0.1131f, 0.1141f}, 
                                                          {0.1211f, 0.1221f, 0.1231f, 0.1241f},
                                                          {0.1311f, 0.1321f, 0.1331f, 0.1341f}}, 
            
                                                         {{0.1112f, 0.1122f, 0.1132f, 0.1142f},
                                                          {0.1212f, 0.1222f, 0.1232f, 0.1242f},
                                                          {0.1312f, 0.1322f, 0.1332f, 0.1342f}}},
                                                         
                                                         {{{0.2111f, 0.2121f, 0.2131f, 0.2141f}, 
                                                          {0.2211f, 0.2221f, 0.2231f, 0.2241f},
                                                          {0.2311f, 0.2321f, 0.2331f, 0.2341f}}, 
            
                                                         {{0.2112f, 0.2122f, 0.2132f, 0.2142f},
                                                          {0.2212f, 0.2222f, 0.2232f, 0.2242f},
                                                          {0.2312f, 0.2322f, 0.2332f, 0.2342f}}}        
                                                        });
        int row = 2;
        int col = 3;
        int z = 1;
        int fourthDim = 1;
        float val = 0.12f;

        instance.add(row, col, z, fourthDim, val);
        float expResult = 0.3542f;                // 0.2342f + 0.12f
        float result = instance.get(row, col, z, fourthDim);         
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testAdd_int_float() {
        Tensor instance = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});
        int idx = 2;
        instance.add(idx, 0.3f);        
        float[] result = instance.getValues();
        float[] expResult = new float[] {0.2f, 0.4f, 0.9f, 0.8f};        
        assertArrayEquals(expResult, result, 1e-7f);
    }

    @Test
    public void testAdd_Tensor() {
        Tensor t1 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.5f});        
        Tensor t2 = new Tensor(new float[] {0.1f, 0.2f, 0.3f, 0.2f});                
        float[] expResult = new float[] {0.3f, 0.6f, 0.9f, 0.7f};
        t1.add(t2);        
        assertArrayEquals(expResult, t1.getValues(), 1e-7f);  
    }

    @Test
    public void testSub_3args() {
        Tensor instance = new Tensor(new float[][] {{0.11f, 0.12f, 0.13f}, {0.21f, 0.22f, 0.23f}});
        int row = 1;
        int col = 2;
        float value = 0.1f;        
        
        instance.sub(row, col, value);                
        float expResult = 0.13f;                
        assertEquals(expResult, instance.get(row, col), 0.0);
    }

    @Test
    public void testSub_4args() {
                                          //  z,r,c          
        Tensor instance = new Tensor(new float[][][] { {{0.111f, 0.121f, 0.131f, 0.141f}, 
                                                        {0.211f, 0.221f, 0.231f, 0.241f},
                                                        {0.311f, 0.321f, 0.331f, 0.341f}}, 
            
                                                        {{0.112f, 0.122f, 0.132f, 0.142f},
                                                         {0.212f, 0.222f, 0.232f, 0.242f},
                                                         {0.312f, 0.322f, 0.332f, 0.342f}}});
        int row = 2;
        int col = 3;
        int z = 1;
        float val = 0.12f;

        instance.sub(row, col, z, val);
        float expResult = 0.222f; // 0.342-0.12                
        float result = instance.get(row, col, z);         
        assertEquals(expResult, result, 0.0);        
    }

    @Test
    public void testSub_5args() {
        Tensor instance = new Tensor(new float[][][][] { {{{0.1111f, 0.1121f, 0.1131f, 0.1141f}, 
                                                          {0.1211f, 0.1221f, 0.1231f, 0.1241f},
                                                          {0.1311f, 0.1321f, 0.1331f, 0.1341f}}, 
            
                                                         {{0.1112f, 0.1122f, 0.1132f, 0.1142f},
                                                          {0.1212f, 0.1222f, 0.1232f, 0.1242f},
                                                          {0.1312f, 0.1322f, 0.1332f, 0.1342f}}},
                                                         
                                                         {{{0.2111f, 0.2121f, 0.2131f, 0.2141f}, 
                                                          {0.2211f, 0.2221f, 0.2231f, 0.2241f},
                                                          {0.2311f, 0.2321f, 0.2331f, 0.2341f}}, 
            
                                                         {{0.2112f, 0.2122f, 0.2132f, 0.2142f},
                                                          {0.2212f, 0.2222f, 0.2232f, 0.2242f},
                                                          {0.2312f, 0.2322f, 0.2332f, 0.2342f}}}        
                                                        });
        int row = 2;
        int col = 3;
        int z = 1;
        int fourthDim = 1;
        float val = 0.12f;

        instance.sub(row, col, z, fourthDim, val);
        float expResult = 0.1142f;                // 0.2342f - 0.12f
        float result = instance.get(row, col, z, fourthDim);         
        assertEquals(expResult, result, 0.0);
    }

    @Test
    public void testSub_Tensor() {
        Tensor t1 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.5f});        
        Tensor t2 = new Tensor(new float[] {0.1f, 0.2f, 0.3f, 0.2f});                
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.3f};
        t1.sub(t2);        
        assertArrayEquals(expResult, t1.getValues(), 1e-7f);  
    }

    @Test
    public void testSub_Tensor_Tensor() {
        Tensor t1 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});        
        Tensor t2 = new Tensor(new float[] {0.1f, 0.2f, 0.3f, 0.2f});                
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.6f};
        Tensor.sub(t1, t2);        
        assertArrayEquals(expResult, t1.getValues(), 0.0f);   
    }

    @Test
    public void testDiv_float() {
        Tensor instance = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});
        instance.div(2.0f);
        float[] result = instance.getValues();
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.4f};        
        assertArrayEquals(expResult, result, 0.0f);
    }

//    @Test
//    public void testDiv_Tensor() {
//        Tensor instance = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});
//        
//        instance.div(2.0f);
//        float[] result = instance.getValues();
//        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.4f};        
//        assertArrayEquals(expResult, result, 0.0f);
//        
//        Tensor mat = null;
//        Tensor instance = null;
//        instance.div(mat);
//        fail("The test case is a prototype.");
//    }

    @Test
    public void testFill_float() {
        float[] values = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        Tensor instance = new Tensor(values);
        instance.fill(-0.7f);
        float[] result = instance.getValues();
        float[] expResult = new float[] {-0.7f, -0.7f, -0.7f, -0.7f, -0.7f};        
        assertArrayEquals(expResult, result, 0.0f);
    }

    @Test
    public void testFill_floatArr_float() {
        float[] array = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        float val = -0.7f;
        Tensor.fill(array, val);
        float[] expResult = new float[] {val, val, val, val, val};        
        assertArrayEquals(expResult, array, 0.0f);
    }

    @Test
    public void testDiv_floatArr_float() {
        float[] array = new float[] {0.2f, 0.4f, 0.6f, 0.8f};        
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.4f};
        Tensor.div(array, 2.0f);
        assertArrayEquals(expResult, array, 0.0f);
    }

    @Test
    public void testSub_floatArr_floatArr() {
        float[] array1 = new float[] {0.2f, 0.4f, 0.6f, 0.8f};        
        float[] array2 = new float[] {0.1f, 0.2f, 0.3f, 0.2f};        
        float[] expResult = new float[] {0.1f, 0.2f, 0.3f, 0.6f};
        Tensor.sub(array1, array2);
        assertArrayEquals(expResult, array1, 0.0f);        
    }

    @Test
    public void testAdd_floatArr_floatArr() {
        float[] array1 = new float[] {0.2f, 0.4f, 0.6f, 0.5f};        
        float[] array2 = new float[] {0.1f, 0.2f, 0.3f, 0.2f};        
        float[] expResult = new float[] {0.3f, 0.6f, 0.9f, 0.7f};
        Tensor.add(array1, array2);
        assertArrayEquals(expResult, array1, 1e-7f);          
    }

    @Test
    public void testCopy_Tensor_Tensor() {
        Tensor src = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});        
        Tensor dest = new Tensor(4); 
        Tensor.copy(src, dest);        
        assertArrayEquals(src.getValues(), dest.getValues(), 0.0f);
    }

    @Test
    public void testCopy_floatArr_floatArr() {
        float[] src = new float[] {0.2f, 0.4f, 0.6f, 0.5f};        
        float[] dest = new float[4];   
        Tensor.copy(src, dest);
        assertArrayEquals(src, dest, 0.0f);
    }

    @Test
    public void testEquals_Object() {
        Tensor t1 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});        
        Tensor t2 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f}); 
        Tensor t3 = new Tensor(new float[] {0.2f, 0.4f, 0.7f, 0.8f});                        
        
        boolean expResult = true;
        boolean result = t1.equals(t2);
        assertEquals(expResult, result);
        
        expResult = false;
        result = t1.equals(t3);
        assertEquals(expResult, result);        
    }

    @Test
    public void testEquals_Tensor_float() {
        Tensor t1 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});        
        Tensor t2 = new Tensor(new float[] {0.2f, 0.4f, 0.6f, 0.8f});                
        Tensor t3 = new Tensor(new float[] {0.2f, 0.4f, 0.7f, 0.8f});                
     
        float delta = 1e-7f;
        boolean expResult = true;
        boolean result = t1.equals(t2, delta);
        assertEquals(expResult, result);
        
        expResult = false;
        result = t1.equals(t3, delta);
        assertEquals(expResult, result);        
    }
    
}
