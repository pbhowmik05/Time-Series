import java.util.Random;
import java.util.ArrayList;



class Main
{
    static void usingneuralnet(SupervisedLearner mlearning, SupervisedLearner nlearning) {
        double initialTime = System.nanoTime();
        Matrix data = new Matrix();
        System.out.println("Main.backPropTest()");
        data.loadARFF("data/hypothyroid.arff");
     //   System.out.println("data load.backPropTest()");	
        Matrix trainingFeatures = new Matrix();
        Matrix trainingLabels = new Matrix();
        Matrix validationFeatures = new Matrix();
        Matrix validationLabels = new Matrix();
        Matrix testFeatures = new Matrix();
        Matrix testLabels = new Matrix();
        Random random = new Random();
        
      //  int nCount = data.valueCount(0);
       // System.out.println("ncount: "+nCount); 
        for(int i = 0; i < data.cols()-1; i++){
            
            trainingFeatures.newColumn(data.valueCount(i));
            validationFeatures.newColumn(data.valueCount(i));
            testFeatures.newColumn(data.valueCount(i));
        }
        
        trainingLabels.newColumn(data.valueCount(trainingFeatures.cols()));
        validationLabels.newColumn(data.valueCount(trainingFeatures.cols()));
        testLabels.newColumn(data.valueCount(trainingFeatures.cols()));
        
        int num_row = data.rows();
      //  System.out.println("row_num   " + num_row);
        double num_train1 = num_row/20.0;
        int num_train = (int) (Math.floor(num_train1))*8; 
        
      //System.out.println("row_train   " + num_train);
        double  num_valid1 = (num_row/20.0);
        int num_valid = (int)(Math.floor(num_valid1))*8;        
        int num_test = (num_row - (num_train+num_valid));        
        //System.out.println("row_test   " + num_test );
        int x = 0, y = 0, z = 0;
     		
                        
        int xx = 0;
          for(int i = 0; i < num_train; i++){
            int xxx = (x*20+(i%8));
      //      System.out.println("ncount: "+pankaj);
            Vec rowdata_train = data.row(xxx);
            double[] Valfeat;
            double[] Vallabel;
            
            Valfeat = trainingFeatures.newRow();
            Vallabel = trainingLabels.newRow();
            for(int j = 0; j < Valfeat.length; j++)
                Valfeat[j] = rowdata_train.vals[j];
            for(int j = 0; j < Vallabel.length; j++)
                Vallabel[j] = rowdata_train.vals[Valfeat.length+j]; 
            if(((i+1)%8)==0 && i!=0)
                x++;
        }
          
        for(int i = 0; i < num_valid; i++){
            int yyy = (y*20+(i%8)+8);
       
            Vec rowdata_valid = data.row(yyy);
            double[] Valfeat_valid;
            double[] Vallabel_valid;
            
            Valfeat_valid = validationFeatures.newRow();
            Vallabel_valid = validationLabels.newRow();
            for(int j = 0; j < Valfeat_valid.length; j++)
                Valfeat_valid[j] = rowdata_valid.vals[j];
            for(int j = 0; j < Vallabel_valid.length; j++)
                Vallabel_valid[j] = rowdata_valid.vals[Valfeat_valid.length+j]; 
            if(((i+1)%8)==0 && i!=0)
                y++;
        }

        int yy = 0;
        int temp = ((num_row/20)*4);
        for(int i = 0; i < num_test; i++){
            
            int zzz;
           
            if(i < temp)
                zzz = (z*20+(i%4)+16);
            else {
                zzz = temp + num_train + num_valid + yy;
                yy++;
            }
          //  System.out.println("ncount: "+zzz);
            Vec rowdata_test = data.row(zzz);
            double[] Valfeat_test;
            double[] Vallabel_test;
            
            Valfeat_test = testFeatures.newRow();
            Vallabel_test = testLabels.newRow();
            for(int j = 0; j < Valfeat_test.length; j++)
                Valfeat_test[j] = rowdata_test.vals[j];
            for(int j = 0; j < Vallabel_test.length; j++)
                Vallabel_test[j] = rowdata_test.vals[Valfeat_test.length+j]; 
            if(((i+1)%4)==0 && i!=0)
                ++z;
        } 
      
        int sizeMinibatch = 10;

        mlearning.train(trainingFeatures, trainingLabels);                
        nlearning.train(trainingFeatures, trainingLabels);
        
 //       System.out.println("Traing done m and n learning\n");
        int rowCountFeat = trainingFeatures.rows();
        
        Vec gradient1 = new Vec(((neuralnet_extended) mlearning).weights.len);
        Vec gradient2 = new Vec(((neuralnet_extended) nlearning).weights.len);
        
	System.out.println("Traing done\n");
	double learning_rate1 = 0.07;
        double learning_rate2 = 0.06;
        double momentum = 0.8;
        double priorError = trainingLabels.cols();
        
        for(int s=0; s<300; s++){    
            for (int i = 0; i < 9; i++) {

                ArrayList<Integer> randLayer = new ArrayList<Integer>();
                int rnd = randLayer.size();

                while(randLayer.size() < rowCountFeat) {    
                    int pointer = random.nextInt(rowCountFeat);
                    boolean retval = randLayer.contains(pointer); 
                    if (!retval==true) {
                        randLayer.add(pointer);
                    }
                }

                for(int p=0; p <trainingFeatures.rows(); p++) {
                    int getIndex = randLayer.get(p);
                    //System.out.println("getIndex = "+ getIndex);
                    Vec vectrainingFeature = trainingFeatures.row(getIndex);
                    Vec vectrainingLabels = trainingLabels.row(getIndex);
                  //  System.out.println("getIndex = "+ vectrainingFeature.len);
                   // System.out.println("getIndex = "+ vectrainingLabels.len); 
                    ((neuralnet_extended) nlearning).refineWeights_momentum(vectrainingFeature, vectrainingLabels, gradient2, learning_rate2, momentum);

                    int reminder = p%sizeMinibatch;
                    double divident = 1.0/sizeMinibatch;
                    if(reminder==0 && p!=0) {
                        gradient1.scale(divident);
                        ((neuralnet_extended) mlearning).refineWeights_batch(gradient1, learning_rate1);
                        gradient1.fill(0);
                    }
                    else ((neuralnet_extended) mlearning).updateGradient_batch(vectrainingFeature, vectrainingLabels, gradient1);
                
                }
              } 

                if(s%10==0){
                                        
                    double errorTrain = mlearning.sumSquaredError(trainingFeatures, trainingLabels);
                    double errorValidation = mlearning.sumSquaredError(validationFeatures, validationLabels);
                    

                    double misclassificationRateTrain1 = mlearning.countMisclassifications(trainingFeatures, trainingLabels);
                    double misclassificationRateTrain2 = nlearning.countMisclassifications(trainingFeatures, trainingLabels);
                    double misclassificationRateValidation1 = mlearning.countMisclassifications(validationFeatures, validationLabels);
                    double misclassificationRateValidation2 = nlearning.countMisclassifications(validationFeatures, validationLabels);
                    
                    double currentTime = System.nanoTime();
                    double elaspedTime = (currentTime-initialTime)/1000000000.0;
                    
                    System.out.println("SSE["+s+"]         ::SSE in Train data set                           :  " +  errorTrain);
                    System.out.println("SSE["+s+"]         ::SSE in Validation data Set                      :  " + errorValidation);
                    System.out.println("Time (s)["+s+"]    ::Elapsed Time                                    :  " + elaspedTime+"s.");
                    System.out.println("Mini Batch["+s+"]  ::Misclassification rate in Training Data Set(%)  :  " +misclassificationRateTrain1);
                    System.out.println("Mini Batch["+s+"]  ::Misclassification rate in Validation Data Set(%):  " +misclassificationRateValidation1);
                    System.out.println("Momentum["+s+"]    ::Misclassification rate in Training Data Set(%)  :  " +misclassificationRateTrain2);
                    System.out.println("Momentum["+s+"]    ::Misclassification rate in Validation Data Set(%):  " +misclassificationRateValidation2);
                    double errorRate = 100.0*((priorError-errorTrain)/priorError);
                    
                    if(errorRate<=1.0 && s>0){
                        System.out.println("Converge.........[error rate is less than 1%]");
                    }
                    priorError = errorTrain;       
                }

 
        }

        double error_missclass = mlearning.countMisclassifications(testFeatures, testLabels);
        System.out.println("Test data set has misscalssification rate :" + error_missclass);
        
    }

    public static void modelTrain(SupervisedLearner mlearning, SupervisedLearner nlearning) {
	//System.out.println("Traing testlearner\n");	
            usingneuralnet(mlearning, nlearning);

	}

    public static void main(String[] args)
    {          
	modelTrain(new neuralnet_extended(), new neuralnet_extended());
                //System.out.println("Traing main\n");
    }
} 