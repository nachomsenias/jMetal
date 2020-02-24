package org.uma.jmetal.algorithm.singleobjective;

import java.util.Comparator;

import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.operator.LocalSearchOperator;
import org.uma.jmetal.operator.impl.localsearch.BasicLocalSearch;
import org.uma.jmetal.operator.impl.mutation.HillClimbingMutation;
import org.uma.jmetal.solution.IntegerSolution;
import org.uma.jmetal.util.comparator.FitnessComparator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import util.random.RandomizerUtils;

import org.uma.jmetal.problem.Problem;

public class HC implements Algorithm<IntegerSolution>{

	/**
	 * Generated.
	 */
	private static final long serialVersionUID = 365695617349156103L;
	
	private int rounds;
	private int iteration;
	private int evals;
	private Problem<IntegerSolution> problem;
	
	private IntegerSolution bestSolution;
	
	public HC(int iteration, int rounds, int evals, Problem<IntegerSolution> problem) {
		this.iteration=iteration;
		this.rounds=rounds;
		this.evals = evals;
		this.problem=problem;
	}

	@Override
	public String getName() {
		return "HC";
	}

	@Override
	public String getDescription() {
		return "Simple HC implementation.";
	}

	@Override
	public void run() {
		bestSolution = null;
		JMetalRandom.getInstance().setSeed(RandomizerUtils.PRIME_SEEDS[iteration]);
		for (int i=0; i<rounds; i++) {
			Comparator<IntegerSolution> comparator = new FitnessComparator<>();
			
			HillClimbingMutation operator = new HillClimbingMutation();
			
			LocalSearchOperator<IntegerSolution> localSearch = 
					new BasicLocalSearch<IntegerSolution>( evals, operator, comparator, problem);
			IntegerSolution solution = problem.createSolution();
			problem.evaluate(solution);
			IntegerSolution improvedSolution = localSearch.execute(solution);
			
			int improvements = localSearch.getNumberOfImprovements();
			if (improvements>0) {
				System.out.println(
					"Finished HC: Number of improvements ("+improvements
					+"). Current number of evaluations ("+i*localSearch.getEvaluations()+").");
			}
			
			if(bestSolution == null || comparator.compare(improvedSolution,bestSolution)==-1) {
				bestSolution = improvedSolution;
			}
		}
	}

	@Override
	public IntegerSolution getResult() {
		return bestSolution;
	}
}