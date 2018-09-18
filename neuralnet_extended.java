public class neuralnet_extended extends NeuralNet {
    
    Imputer mImputer1; 
    Imputer mImputer2;
    NomCat mNomCat1;
    NomCat  mNomCat2;
    Normalizer mNormalizer1;
    Normalizer  mNormalizer2;
            
    int imputer_feat_size;
    int imputer_label_size;
    int nomcat_feat_size;
    int nomcat_label_size;
    
    @Override
    void train(Matrix feature, Matrix label) {
    
        mImputer1 = new Imputer();
        mImputer1.train(feature);
        mImputer2 = new Imputer();
        mImputer2.train(label);
        
        imputer_feat_size = feature.cols();
        imputer_label_size = label.cols();
        
        mNomCat1 = new NomCat();
        mNomCat1.train(feature);
        
        mNomCat2 = new NomCat();       
        mNomCat2.train(label);
        
        nomcat_feat_size = mNomCat1.outputTemplate().cols();
        nomcat_label_size = mNomCat2.outputTemplate().cols();
        
        Matrix nomCat_feature = new Matrix();
        Matrix nomCat_label = new Matrix();
        
        int [] feature_return = mNomCat1.referenceValue();
        int [] label_return = mNomCat2.referenceValue();
        int featLen = feature_return.length;
        int labelLen = label_return.length;
        
        for(int element1: feature_return) {           
            if(element1 ==-1)
                nomCat_feature.newColumn(2);
            else
                nomCat_feature.newColumn(0);
        }
        
        for(int element2: label_return) {           
            if(element2 ==-1)
                nomCat_label.newColumn(2);
            else
                nomCat_label.newColumn(0);
        }
        
//normalize for feature
        double [] featureLine1 = nomCat_feature.newRow();   //col_min value
        double [] featureLine2 = nomCat_feature.newRow();   //col_maxvalue
        double [] labelLine1 =nomCat_label.newRow();
        double [] labelLine2 =nomCat_label.newRow();
//Feature calculation                
        for(int i = 0; i<featLen; i++) {
            featureLine1[i]=((feature_return[i] ==-1)? Matrix.UNKNOWN_VALUE: feature.columnMin(feature_return[i]));
            featureLine2[i]=((feature_return[i] ==-1)? Matrix.UNKNOWN_VALUE: feature.columnMax(feature_return[i]));               
        }
//Label calculation               
        for(int i = 0; i<labelLen; i++) {
            labelLine1[i]= ((label_return[i] ==-1)? Matrix.UNKNOWN_VALUE:label.columnMin(label_return[i]));
            labelLine2[i]= ((label_return[i] ==-1)? Matrix.UNKNOWN_VALUE:label.columnMax(label_return[i]));            
        }
        mNormalizer1 = new Normalizer();
        mNormalizer2 = new Normalizer();
        
        mNormalizer1.train(nomCat_feature);
        mNormalizer2.train(nomCat_label);
        
        super.train(mNormalizer1.outputTemplate(), mNormalizer2.outputTemplate());
        

    }
    
        double[] transform(Vec in, int selector){
            
            if(selector==1) {
                double[] imputerTransform_feat = new double [in.vals.length];
                mImputer1.transform(in.vals, imputerTransform_feat);
                double[] nomcatTransform_feat = new double[nomcat_feat_size];
                //System.out.println("nomcatTransform_feat = "+ nomcatTransform_feat.length+ "    "+imputerTransform_feat.length);
                
                mNomCat1.transform(imputerTransform_feat, nomcatTransform_feat);
                double[] normalizerTransform_feat = new double[nomcat_feat_size];
                mNormalizer1.transform(nomcatTransform_feat, normalizerTransform_feat);

                return normalizerTransform_feat;
            }
            ///////////////////////////////////////////////
            else {
                double[] imputerTransform_label = new double [in.vals.length];
                mImputer2.transform(in.vals, imputerTransform_label);           
                double[] nomcatTransform_label = new double[nomcat_label_size];           
                mNomCat2.transform(imputerTransform_label, nomcatTransform_label);            
                double[] normalizerTransform_label = new double[nomcat_label_size];            
                mNormalizer2.transform(nomcatTransform_label, normalizerTransform_label);

                return normalizerTransform_label;
            }
        }
        double[] untransform(Vec out, int selector){
        
            if(selector==1){
                double[] normalizerTransform_feat = new double[out.vals.length];
                mNormalizer1.untransform(out.vals, normalizerTransform_feat);
                double[] nomcatTransform_feat = new double[imputer_feat_size];
                mNomCat1.untransform(normalizerTransform_feat, nomcatTransform_feat);
                double[] imputerTransform_feat = new double[imputer_feat_size];
                mImputer1.untransform(nomcatTransform_feat, imputerTransform_feat);

                return imputerTransform_feat;
            } 
            else {
                double[] normalizerTransform_label = new double[out.vals.length];           
                mNormalizer2.untransform(out.vals, normalizerTransform_label);            
                double[] nomcatTransform_label = new double[imputer_label_size];            
                mNomCat2.untransform(normalizerTransform_label, nomcatTransform_label);            
                double[] imputerTransform_label = new double[imputer_label_size];            
                mImputer2.untransform(nomcatTransform_label, imputerTransform_label);

                return imputerTransform_label;
            }
        }
    
    @Override
    Vec predict(Vec in) {
        double[] normalizerTransform_feat = transform(in,1);
        Vec Vec_normalizerTransform_feat = new Vec (normalizerTransform_feat);
        Vec out = super.predict(Vec_normalizerTransform_feat);
        
        double[] imputerTransform_label = untransform(out, 2);
        Vec Vec_imputerTransform_label = new Vec (imputerTransform_label);
        
        return Vec_imputerTransform_label;      
    }    
    @Override
        void refineWeights(Vec x, Vec y, double learningRate) {
            double[] normalizerTransform_feat = transform(x,1);
            double[] normalizerTransform_label = transform(y, 2);
            Vec Vec_normalizerTransform_feat = new Vec (normalizerTransform_feat);
            Vec Vec_normalizerTransform_label = new Vec (normalizerTransform_label);
            super.refineWeights(Vec_normalizerTransform_feat, Vec_normalizerTransform_label, learningRate);
        }
        
    @Override
    void updateGradient_batch(Vec x, Vec y, Vec gradient) {   
            double[] normalizerTransform_feat = transform(x,1);
            double[] normalizerTransform_label = transform(y,2);
            Vec Vec_normalizerTransform_feat = new Vec (normalizerTransform_feat);
            Vec Vec_normalizerTransform_label = new Vec (normalizerTransform_label);
            
            super.updateGradient_batch(Vec_normalizerTransform_feat, Vec_normalizerTransform_label, gradient);
    } 
        
    @Override
        void refineWeights_momentum(Vec x, Vec y, Vec gradient, double learningRate, double momentum){
            
            double[] normalizerTransform_feat = transform(x,1);
            double[] normalizerTransform_label = transform(y,2);
            Vec Vec_normalizerTransform_feat = new Vec (normalizerTransform_feat);
            Vec Vec_normalizerTransform_label = new Vec (normalizerTransform_label);
            
            super.refineWeights_momentum(Vec_normalizerTransform_feat, Vec_normalizerTransform_label, gradient, learningRate, momentum);        
        }
}