package org.uma.jmetal.algorithm.impl;

import java.util.List;

import org.uma.jmetal.problem.Problem;

public abstract class AbstractMemeticAlgorithm<S, R> 
		extends AbstractEvolutionStrategy<S, R> {

	public AbstractMemeticAlgorithm(Problem<S> problem) {
		super(problem);
	}

	/**
	 * Generated.
	 */
	private static final long serialVersionUID = -1703003424101169286L;

	protected abstract List<S> localImprovement(List<S> population);

	@Override
	public void run() {
		List<S> offspringPopulation;
		List<S> matingPopulation;

		population = createInitialPopulation();
		population = evaluatePopulation(population);
		initProgress();
		while (!isStoppingConditionReached()) {
			matingPopulation = selection(population);
			offspringPopulation = reproduction(matingPopulation);
			offspringPopulation = evaluatePopulation(offspringPopulation);
			offspringPopulation = localImprovement(offspringPopulation);
			population = replacement(population, offspringPopulation);
			updateProgress();
		}
	}
}
