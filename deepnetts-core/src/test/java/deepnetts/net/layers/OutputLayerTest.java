package deepnetts.net.layers;


import deepnetts.net.loss.LossType;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import deepnetts.util.WeightsInit;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class OutputLayerTest {
    

    @Test
    public void testForward() {
        
        RandomGenerator.getDefault().initSeed(123);         // initialize weights using specified random seed
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer (output for previous layer)
        Tensor weights = new Tensor(5, 10); 
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]" 

       Tensor expectedOutputs = new Tensor( 0.51053022f, 0.59142921f, 0.52648754f, 0.56102458f, 0.54380692f, 0.58635918f, 0.54137987f, 0.41367945f, 0.52289978f, 0.4961883f );
         
        // create prev fc layer with 5 outputs
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5);        
        prevLayer.setOutputs(input); // and set its ouput that will be used as input for next layer
                
        // create instance of layer to test
        OutputLayer instance = new OutputLayer(10);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights structures
        instance.setWeights(weights); // set weights values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // run forward pass
        instance.forward();
        
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }

    /**
     * These values are for backward pass for MSE loss function.
     * TODO: Fix this tests, it breaks
     */
    @Test
    public void testBackwardMseLoss() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights (same effect as line above)
        
        // input vector for this layer
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // weights from previous layer
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
        Tensor outputErrors = new Tensor(10);
        outputErrors.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        OutputLayer instance = new OutputLayer(10);
        instance.setLossType(LossType.MEAN_SQUARED_ERROR);
        instance.setPrevLayer(prevLayer);               
        instance.init();
        instance.setWeights(weights);
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        instance.forward(); // derivatives are calculated using outputs | outputs : "[0.51053023, 0.59142923, 0.5264875, 0.5610246, 0.5438069, 0.5863592, 0.5413798, 0.41367948, 0.52289975, 0.4961883]"          
        instance.setOutputErrors(outputErrors.getValues());

        instance.backward();
                
        Tensor result = instance.getDeltas(); //   [0.01052711, 0.08937729, 0.026437959, 0.060416743, 0.043582395, 0.08463131, 0.04119066, -0.084595, 0.022867741, -0.0038115513]
        Tensor expResult = new Tensor(10); // [0.01052711  0.08937729  0.02643796   0.06041675   0.0435824    0.08463131  0.04119066  -0.084595  0.02286774   -0.00381155] from num py test
        expResult.setValues( 0.01052711f,  0.08937729f,  0.02643796f,  0.06041675f,  0.0435824f,   0.08463131f,  0.04119066f, -0.084595f, 0.02286774f, -0.00381155f);
        
        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-8f);
    }

    @Test
    public void testApplyWeightChanges() {
        OutputLayer instance = new OutputLayer(10);
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5);
        instance.setPrevLayer(prevLayer);
        instance.init();
        Tensor weights = new Tensor(5, 10);

        WeightsInit.initSeed(123);
        WeightsInit.uniform(weights.getValues(), 5);    
        Tensor biases = new Tensor(10); // 
        WeightsInit.randomize(weights.getValues()); //  [-0.2885946, -0.023120344, 0.114212096, -0.35789996, -0.46654212, -0.060512245, 0.18889612, -0.38798183, 0.18411279, 0.34890008, -0.4247411, -0.3433265, 0.44680566, -0.09958029, -0.063123405, 0.43927592, 0.19381553, -0.083636045, 0.17639846, 0.32855767, 0.12878358, 0.1629799, 0.07470411, -0.46918643, -0.11202836, 0.2753712, -0.31087178, -0.08334535, -0.47327846, -0.3146028, -0.46844327, 0.112583816, 0.33040798, -0.10281438, 0.24008608, 0.32558167, 0.44147235, 0.32505035, 0.32593954, -0.17731398, 0.014207661, 0.28614485, 0.09407586, 0.123054445, -0.27172554, -0.14735949, -0.19683117, -0.33119786, 0.19504476, 0.23377019]
        WeightsInit.randomize(biases.getValues()); //   [-0.07173675, -0.06975782, 0.46547735, -0.06449604, 0.22543085, -0.25612664, -0.16484225, -0.21565866, 0.45828927, -0.13396758]
        instance.setWeights(weights);
        instance.setBiases(biases.getValues());        
        instance.setDeltas(new Tensor(10));
        instance.deltaWeights = new Tensor(5, 10);
        WeightsInit.randomize(instance.deltaWeights.getValues()); // [-0.014675736, -0.20824462, -0.2544266, -0.29433697, 0.19522274, -0.042135, -0.2805665, 0.44587213, -0.38881636, 0.2882418, 0.06629634, 0.011203349, 0.014710844, 0.23828048, -0.34540945, -0.16317445, 0.052065372, 0.35782564, -0.2368347, 0.16372514, -0.21912044, 0.32790893, -0.08429372, -0.347282, -0.33204705, -0.25469518, 0.33288032, -0.22248, -0.19762617, -0.30643207, -0.05354631, -0.46328926, 0.23842907, 0.25631714, -0.4825107, 0.0952881, -0.32452244, -0.014563918, 0.48131067, -0.3766222, 0.44226605, -0.12013543, 0.35074377, -0.29712552, 0.259381, 0.49666756, -0.43016553, -0.40166676, -0.012900829, 0.3826325]
        WeightsInit.randomize(instance.deltaBiases); // [-0.32544816, -0.42210114, 0.46139675, 0.44991916, 0.11481196, 0.48068362, 0.2159949, 0.23206848, -0.41850775, 0.28163642]
                
        instance.applyWeightChanges();        
        Tensor expectedWeights = new Tensor(-0.30327034f, -0.23136496f, -0.1402145f, -0.65223693f, -0.27131938f, -0.10264724f, -0.09167038f,  0.0578903f,  -0.20470357f,  0.63714188f, -0.35844476f, -0.33212315f, 0.4615165f,   0.13870019f, -0.40853286f,  0.27610147f,  0.2458809f,   0.2741896f, -0.06043624f,  0.49228281f, -0.09033686f,  0.49088883f, -0.00958961f, -0.81646843f, -0.44407541f,  0.02067602f,  0.02200854f, -0.30582535f, -0.67090463f, -0.62103487f, -0.52198958f, -0.35070544f,  0.56883705f,  0.15350276f, -0.24242462f,  0.42086977f, 0.11694991f,  0.31048643f,  0.80725021f, -0.55393618f,  0.45647371f,  0.16600942f, 0.44481963f, -0.17407107f, -0.01234454f,  0.34930807f, -0.6269967f,  -0.73286462f, 0.18214393f,  0.61640269f);        
        Tensor expectedBiases = new Tensor(-0.39718491f, -0.49185896f, 0.9268741f, 0.38542312f, 0.34024281f, 0.22455698f, 0.05115265f, 0.01640982f, 0.03978152f, 0.14766884f);
                
        assertArrayEquals(weights.getValues(), expectedWeights.getValues(), 1e-7f);
        assertArrayEquals(biases.getValues(), expectedBiases.getValues(), 1e-7f);  
    }
    
}
