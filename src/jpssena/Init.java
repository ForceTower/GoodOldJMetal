package jpssena;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.nsgaII.NSGAII;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import jmetal.util.JMException;
import weka.core.Instances;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class Init {
    public static void main(String[] args) throws IOException, JMException, ClassNotFoundException {
        HashMap parameters ; // Operator parameters

        File file = new File("C:\\Users\\joaop\\Desktop\\banana\\banana-10-1tra.dat");
        File fixed = DatFixer.fixDatFormat(file);
        Instances instances = new Instances(new FileReader(fixed));
        if (instances.classIndex() == -1)
            instances.setClassIndex(instances.numAttributes() - 1);

        Problem problem = new LearnSelectInstances(instances);

        Algorithm algorithm = new NSGAII(problem);

        algorithm.setInputParameter("populationSize",100);
        System.out.println("Population size.................: " + algorithm.getInputParameter("populationSize").toString());
        algorithm.setInputParameter("maxEvaluations",1000);
        System.out.println("Max evaluations.................: " + algorithm.getInputParameter("maxEvaluations").toString());

        parameters = new HashMap();
        parameters.put("probability", 0.9);
        Operator crossover = CrossoverFactory.getCrossoverOperator("HUXCrossover", parameters);

        parameters = new HashMap();
        parameters.put("probability", 0.2);
        Operator mutation = MutationFactory.getMutationOperator("BitFlipMutation", parameters);

        parameters = new HashMap();
        Operator selection = SelectionFactory.getSelectionOperator("BinaryTournament2", parameters) ;

        algorithm.addOperator("crossover", crossover);
        algorithm.addOperator("mutation", mutation);
        algorithm.addOperator("selection", selection);

        long initTime = System.currentTimeMillis();

        SolutionSet population = algorithm.execute();

        double estimatedTime = System.currentTimeMillis() - initTime;
        double aux = estimatedTime * 0.001; // converted in seconds
        double timeSelecInstances = aux * 0.0167;  // converted in minutes

        System.out.println("\nTime to select instances.: " + timeSelecInstances + " minutes.");
        System.out.println("Number of solutions......: " + population.size());
    }
}
