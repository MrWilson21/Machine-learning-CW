import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Main
{

    public static void main(String[] args)
    {
        Instances train = WekaTools.loadClassificationData("bank_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("bank_TEST.arff");

        EnhancedLinearPerceptron lp = new EnhancedLinearPerceptron();

        lp.maxIterations = 10;
        lp.biasTerm = 0;
        lp.normaliseAttributes = true;
        lp.learningRate = 1;
        lp.modelSelection = true;

        System.out.println(WekaTools.accuracy(lp, test, train));
    }
}
