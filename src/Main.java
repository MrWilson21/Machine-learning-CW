import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Main
{

    public static void main(String[] args)
    {
        Instances train = WekaTools.loadClassificationData("bank_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("bank_TEST.arff");

        LinearPerceptronEnsemble lp = new LinearPerceptronEnsemble();
        EnhancedLinearPerceptron lp2 = new EnhancedLinearPerceptron();

        try
        {
            lp.buildClassifier(test);
            lp2.buildClassifier(test);
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        //lp.distributionForInstance(train.instance(0));
        System.out.println(WekaTools.accuracy(lp, test, train));
        System.out.println(WekaTools.accuracy(lp2, test, train));
    }
}
