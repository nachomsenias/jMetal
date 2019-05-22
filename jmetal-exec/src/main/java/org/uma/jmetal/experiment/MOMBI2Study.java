package org.uma.jmetal.experiment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.mombi.MOMBI2;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT1;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT2;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT3;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.experiment.Experiment;
import org.uma.jmetal.util.experiment.ExperimentBuilder;
import org.uma.jmetal.util.experiment.component.ExecuteAlgorithms;
import org.uma.jmetal.util.experiment.util.ExperimentAlgorithm;
import org.uma.jmetal.util.experiment.util.ExperimentProblem;

public class MOMBI2Study {

	private static final int runs = 30;

	public static void main(String[] args) throws IOException {

		if (args.length != 1) {
			throw new JMetalException("Missing argument: experimentBaseDirectory");
		}
		String experimentBaseDirectory = args[0];

		List<ExperimentProblem<DoubleSolution>> problemList = new ArrayList<>();
		problemList.add(new ExperimentProblem<>(new ZDT1()));
		problemList.add(new ExperimentProblem<>(new ZDT2()));
		problemList.add(new ExperimentProblem<>(new ZDT3()));

		List<ExperimentAlgorithm<DoubleSolution, List<DoubleSolution>>> algorithmList = configureAlgorithmList(
				problemList);

		Experiment<DoubleSolution, List<DoubleSolution>> experiment = 
				new ExperimentBuilder<DoubleSolution, List<DoubleSolution>>("MOMBI2Study")
						.setAlgorithmList(algorithmList)
						.setProblemList(problemList)
						.setExperimentBaseDirectory(experimentBaseDirectory)
						.setOutputParetoFrontFileName("FUN")
						.setOutputParetoSetFileName("VAR")
						.setReferenceFrontDirectory("/pareto_fronts")
						.setIndependentRuns(runs)
						.build();

		new ExecuteAlgorithms<>(experiment).run();
	}

	private static List<ExperimentAlgorithm<DoubleSolution, List<DoubleSolution>>> 
				configureAlgorithmList(List<ExperimentProblem<DoubleSolution>> problemList) {
		List<ExperimentAlgorithm<DoubleSolution, List<DoubleSolution>>> algorithms = new ArrayList<>();

		int maxEvaluations = 10000;
		int populationSize = 100;
		int generations = maxEvaluations / populationSize;
		
		for (int i = 0; i < problemList.size(); i++) {

//			// XXX FIXED CODE
//			Algorithm<List<DoubleSolution>> algorithm = null;
//			
//			for (int mc=0; mc<runs; mc++) {
//				algorithm = new MOMBI2<DoubleSolution>(
//						problemList.get(i).getProblem(),
//						generations, 
//						new SBXCrossover(1.0, 5),
//						new PolynomialMutation(
//								1.0 / problemList.get(i).getProblem().getNumberOfVariables(),
//								10.0), 
//						new BinaryTournamentSelection<DoubleSolution>(),
//						new SequentialSolutionListEvaluator<DoubleSolution>(), 
//						"resources/mombi2-weights/weight/weight_02D_152.sld");
//				((MOMBI2<DoubleSolution>)algorithm)
//					.setMaxPopulationSize(populationSize);
//				algorithms.add(new ExperimentAlgorithm<>(algorithm, "MOMBI2",
//						problemList.get(i),mc));
//			}
//			// XXX END -- FIXED CODE
			
			
			// XXX ERROR CODE
			Algorithm<List<DoubleSolution>> algorithm = 
					new MOMBI2<DoubleSolution>(
						problemList.get(i).getProblem(),
						generations, 
						new SBXCrossover(1.0, 5),
						new PolynomialMutation(
								1.0 / problemList.get(i).getProblem().getNumberOfVariables(),
								10.0), 
						new BinaryTournamentSelection<DoubleSolution>(),
						new SequentialSolutionListEvaluator<DoubleSolution>(), 
						"resources/mombi2-weights/weight/weight_02D_152.sld");
			((MOMBI2<DoubleSolution>)algorithm).setMaxPopulationSize(populationSize);
			
			for (int mc=0; mc<runs; mc++) {
				algorithms.add(
						new ExperimentAlgorithm<>(algorithm, "MOMBI2",problemList.get(i),mc)
						);
			}
			// XXX END -- ERROR CODE
		}
		
		return algorithms;
	}
}
