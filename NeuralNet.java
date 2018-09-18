
//import com.sun.xml.internal.bind.v2.runtime.output.MTOMXmlOutput;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author pankaj
 */
public class NeuralNet extends SupervisedLearner {

    ArrayList<Layer> mLayers = new ArrayList<>();
    //ArrayList<Vec> mWeights = new ArrayList<>();
    Vec weights;
    int layerWeightSize;
    @Override
    String name() {
        return "NeuralNet";
    }

    @Override
    void train(Matrix features, Matrix labels) {
        //  LayerLinear layer = new LayerLinear(features.cols(), labels.cols());
        System.out.println("Train initiating\n");
        layerWeightSize = 0;
        int m_size = 0;
        int b_size = 0;

        LayerLinear L1 = new LayerLinear(33, 100);
        LayerTanh L2 = new LayerTanh(100);
        LayerLinear L3 = new LayerLinear(100, 4);
        LayerTanh L4 = new LayerTanh(4);
        //LayerLinear L5 = new LayerLinear(30, 10);
        //LayerTanh L6 = new LayerTanh(10);

        mLayers.add(L1);
        mLayers.add(L2);
        mLayers.add(L3);
        mLayers.add(L4);
        //mLayers.add(L5);
       // mLayers.add(L6);

        for (int i = 0; i < mLayers.size(); i = i + 2) {
            m_size = mLayers.get(i).mInput_size * mLayers.get(i).mOutput_size;
            b_size = mLayers.get(i).mOutput_size;
            layerWeightSize += (m_size + b_size);
        }
        //System.out.println("layerWeightSize" + layerWeightSize);
        weights = new Vec(layerWeightSize);
        initWeights();
    }

    void initWeights() {

        int wright_index = 0;

        Random random = new Random();
        for (int i = 0; i < 4; i = i + 2) {
            int layer_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);

            for (int j = 0; j < layer_len; j++) {
                double kk= Math.max(0.03, 1.0 / mLayers.get(i).mInput_size) * random.nextGaussian();
                weights.set(j+wright_index, kk);
              //  System.out.println("NeuralNet.initWeights()");
            }
            wright_index += layer_len;
        }
    }

 
    @Override
    Vec predict(Vec in) {        
        Vec temp = new Vec(in, 0, in.len);
        int index = 0;
        for (int i = 0; i < mLayers.size(); i++) {
            Vec out_layer = new Vec(0);
            if ((i % 2) == 0) {
                int nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                //System.out.println("Layer length: " + nWeight_len);
                //System.out.println("Weight_length: " + l);
                out_layer = new Vec(weights, index, nWeight_len);
                //System.out.println("out_layer size :  " +out_layer.len);
                index = index + nWeight_len;
            }
            mLayers.get(i).activate(out_layer, temp);
            temp = new Vec(mLayers.get(i).activation, 0, mLayers.get(i).activation.len);  
            //System.out.println("temp2 size :  " +temp.len);            
        }
        return temp;
    }

    void backprop(Vec weights, Vec target) {
        Vec layer_blame = new Vec(target, 0, target.len);
        Vec y_actual = mLayers.get(mLayers.size() - 1).activation;

        //System.out.println("y_value1 :  " +y_value1);  
        y_actual.scale(-1.0);

        layer_blame.add(y_actual);
        int back_index = weights.len;
        for (int i = mLayers.size() - 1; i >= 0; i--) {
            //System.out.println("loop count :  " +i); 
            if(i>0) {
            mLayers.get(i).blame.copy(layer_blame);
            Vec iweight = new Vec(0);
            int mod = i % 2;
            if (mod == 0) {
                int nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                  back_index = back_index - nWeight_len;
                iweight = new Vec(weights, back_index, nWeight_len);
            }
            layer_blame = new Vec(mLayers.get(i - 1).mOutput_size);
            mLayers.get(i).backProp(iweight, layer_blame);
             }
            else
                mLayers.get(i).blame.copy(layer_blame);
        }

    }

    void updateGradient(Vec x, Vec gradient) {
        int grad_index = 0;

        for (int i = 0; i < mLayers.size(); i++) {
            int grad_size = 0;
            int mod = i % 2;
            //System.out.println("mod value :  " +mod);
            if (mod == 0) 
                grad_size = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);            

            Vec Grad_layer = new Vec(grad_size);

            if (i > 0) mLayers.get(i).updateGradient(mLayers.get(i - 1).activation, Grad_layer); 
            else  mLayers.get(i).updateGradient(x, Grad_layer);
                       
            for (int k = 0; k < grad_size; k++) {
                gradient.set(grad_index + k, Grad_layer.get(k));
            }
            if (mod == 0) {  
                grad_index += grad_size;
            }
            //System.out.println("updateGradient done\n");
        }
    }

    void refineWeights(Vec x, Vec y, double learningRate) {
        predict(x);
        backprop(weights, y);
        Vec gradient = new Vec(weights.len);
        updateGradient(x, gradient);
        weights.addScaled(learningRate, gradient);
     //   System.out.println("NeuralNet.refineWeights()");
    }
    
    void calculateFiniteDifferencing(Vec x) {
        double h = 0.0000001f;
        for (int r = 0; r<4; r++) {

        }
    }
    
    //----------Added code for assignment 3--------------
    Vec predict_update(Vec in) {        
        Vec temp = new Vec(in, 0, in.len);
        int index = 0;
        for (int i = 0; i < mLayers.size(); i++) {
            Vec out_layer = new Vec(0);
            if ((i % 2) == 0) {
                int nWeight_len = mLayers.get(i).mOutput_size + (mLayers.get(i).mOutput_size * mLayers.get(i).mInput_size);
                out_layer = new Vec(weights, index, nWeight_len);
                index = index + nWeight_len;
            }
            mLayers.get(i).activate(out_layer, temp);
            temp = new Vec(mLayers.get(i).activation, 0, mLayers.get(i).activation.len);            
        }
        return temp;
    }
    
    void updateGradient_batch(Vec x, Vec y, Vec gradient) {
        predict_update(x);
        backprop(weights, y);
        Vec grad_batch = new Vec(weights.len);
        updateGradient(x, grad_batch);
        gradient.add(grad_batch);
    }
    
    void refineWeights_batch(Vec gradient, double learningRate){
        weights.addScaled(learningRate, gradient);
    }
    
    void refineWeights_momentum(Vec x, Vec y, Vec gradient, double learningRate, double momentum){
        predict_update(x);
        backprop(weights, y);
        Vec grad_batch = new Vec(weights.len);
        updateGradient(x, grad_batch);
        grad_batch.scale(learningRate);
        grad_batch.addScaled(momentum, gradient);
        weights.add(grad_batch);
        gradient.copy(grad_batch);
    }
}
