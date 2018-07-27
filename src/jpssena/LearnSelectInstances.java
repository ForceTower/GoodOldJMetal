package jpssena;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.encodings.solutionType.BinarySolutionType;
import jmetal.encodings.variable.Binary;
import jmetal.util.JMException;
import mgpires.solutionType.ArrayBinarySolutionType;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.util.BitSet;

public class LearnSelectInstances extends Problem {
    private Instances samples;

    @Override
    public void evaluate(Solution solution) throws JMException {
        Binary sol = (Binary) solution.getDecisionVariables()[0];
        BitSet bitSet = sol.bits_;

        int selected = 0;
        Instances instances = new Instances(samples);

        for (int i = bitSet.length() - 1; i >= 0; i--) {
            if (bitSet.get(i)) {
                selected++;
            } else {
                instances.remove(i);
            }
        }

        double reduction = (samples.numInstances() - selected) / (double)samples.numInstances();
        double accuracy = 0;
        IBk knn = new IBk(5);

        try {
            Evaluation evaluation = new Evaluation(instances);
            knn.buildClassifier(instances);
            //Classify the full into the few
            evaluation.evaluateModel(knn, samples);
            //The number of correct answers
            accuracy = evaluation.correct();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        accuracy = accuracy / samples.numInstances();

        solution.setObjective(0, -1 * accuracy);
        solution.setObjective(1, -1 * reduction);
    }

    LearnSelectInstances(Instances instances) {
        problemName_= "Learn Select Instances";
        numberOfConstraints_ = 0;
        numberOfObjectives_  = 2;

        samples = new Instances(instances);
        numberOfVariables_ = instances.numAttributes();
        solutionType_ = new BinarySolutionType(this);
    }
}
