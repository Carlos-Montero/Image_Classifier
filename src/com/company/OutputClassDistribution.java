
package com.company;


import weka.classifiers.trees.J48;
//import weka.classifiers.naive;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;


public class OutputClassDistribution {

    /**
     * Expects two parameters: training file and test file.
     *
     * @param args	the commandline arguments
     * @throws Exception	if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // load data
        Instances train = DataSource.read("C:\\Users\\Carlos\\Desktop\\Image_Classifier\\indice.arff");  //entrenar
        train.setClassIndex(train.numAttributes() - 1);
        Instances test = DataSource.read("C:\\Users\\Carlos\\Desktop\\Image_Classifier\\indice.arff");    //testear
        test.setClassIndex(test.numAttributes() - 1);
        if (!train.equalHeaders(test))
            throw new IllegalArgumentException(
                    "Train and test set are not compatible: " + train.equalHeadersMsg(test));


        //APLICAR EL FILTRO

        













        // train classifier
        J48 cls = new J48();
        //naive cls = new naive();
        cls.buildClassifier(train);

        // output predictions
        System.out.println("# - actual - predicted - error - distribution");
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = cls.classifyInstance(test.instance(i));
            double[] dist = cls.distributionForInstance(test.instance(i));
            System.out.print((i+1));
            System.out.print(" - ");
            System.out.print(test.instance(i).toString(test.classIndex()));
            System.out.print(" - ");
            System.out.print(test.classAttribute().value((int) pred));
            System.out.print(" - ");
            if (pred != test.instance(i).classValue())
                System.out.print("yes");
            else
                System.out.print("no");
            System.out.print(" - ");
            System.out.print(Utils.arrayToString(dist));
            System.out.println();
        }
    }
}
