package org.uma.jmetal.algorithm.singleobjective.evolutionstrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.commons.math.stat.StatUtils;
import org.uma.jmetal.algorithm.impl.AbstractMemeticAlgorithm;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.util.CMAESUtils;
import org.uma.jmetal.operator.LocalSearchOperator;
import org.uma.jmetal.operator.impl.localsearch.BasicLocalSearch;
import org.uma.jmetal.operator.impl.mutation.HillClimbingMutation;
import org.uma.jmetal.problem.IntegerProblem;
import org.uma.jmetal.problem.impl.AbstractIntegerProblem;
import org.uma.jmetal.solution.IntegerSolution;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.comparator.FitnessComparator;
import org.uma.jmetal.util.comparator.ObjectiveComparator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

/**
 * Class implementing the CMA-ES algorithm
 */
@SuppressWarnings("serial")
public class IntCovarianceMatrixAdaptationEvolutionStrategy
		extends AbstractMemeticAlgorithm<IntegerSolution, IntegerSolution> {
	private Comparator<IntegerSolution> comparator;
	private int lambda;
	private int evaluations;
	private int maxEvaluations;
	private double[] typicalX;

	private final static double UNDEFINED = -1; 
	/**
	 * CMA-ES state variables
	 */

	// Distribution mean and current favorite solution to the optimization
	// problem
	private double[] distributionMean;

	// coordinate wise standard deviation (step size)
	private double sigma;

	// Symmetric and positive definitive covariance matrix
	private double[][] c;

	// Evolution paths for c and sigma
	private double[] pathsC;
	private double[] pathsSigma;

	/*
	 * Strategy parameter setting: Selection
	 */

	// number of parents/points for recombination
	private int mu;

	private double[] weights;
	private double muEff;

	/*
	 * Strategy parameter setting: Adaptation
	 */

	// time constant for cumulation for c
	private double cumulationC;

	// t-const for cumulation for sigma control
	private double cumulationSigma;

	// learning rate for rank-one update of c
	private double c1;

	// learning rate for rank-mu update
	private double cmu;

	// damping for sigma
	private double dampingSigma;

	/*
	 * Dynamic (internal) strategy parameters and constants
	 */

	// coordinate system
	private double[][] b;

	// diagonal D defines the scaling
	private double[] diagD;

	// c^1/2
	private double[][] invSqrtC;

	// track update of b and c
	private int eigenEval;

	private double chiN;

	private IntegerSolution bestSolutionEver = null;

	private Random rand;
	
	private int restarts= 0;
	
	private boolean hc = false;
	
	public final static double pLS = 0.0625;

	/** Constructor */
	private IntCovarianceMatrixAdaptationEvolutionStrategy(
			IntCMAESBuilder builder, long seed) {
		super(builder.problem);
		this.lambda = builder.lambda;
		//Additional fields
		mu = (int) Math.floor(lambda * builder.mu);
		cumulationSigma = builder.cs;
		cumulationC = builder.cc;
		c1 = builder.ccov;
		dampingSigma = builder.damps;
		
		this.maxEvaluations = builder.maxEvaluations;
		this.typicalX = builder.typicalX;
		this.sigma = builder.sigma;

		this.hc = builder.hc;
		
		rand = new Random(seed);
		comparator = new ObjectiveComparator<IntegerSolution>(0);

		initializeInternalParameters();
	}

	/* Getters */
	public int getLambda() {
		return lambda;
	}

	public int getMaxEvaluations() {
		return maxEvaluations;
	}

	/**
	 * Buider class
	 */
	public static class IntCMAESBuilder {
		private static final int DEFAULT_LAMBDA = 10;
		private static final int DEFAULT_MAX_EVALUATIONS = 1000000;
		private static final double DEFAULT_SIGMA = 0.3;

		private IntegerProblem problem;
		private int lambda;
		private int maxEvaluations;
		private double[] typicalX;
		private double sigma;
		// Newly added
		private double mu = -1;
		private double cs = -1;
		private double cc = -1;
		private double ccov = -1;
		private double damps = -1;
		
		private boolean hc = false;

		public IntCMAESBuilder(IntegerProblem problem) {
			this.problem = problem;
			lambda = DEFAULT_LAMBDA;
			maxEvaluations = DEFAULT_MAX_EVALUATIONS;
			sigma = DEFAULT_SIGMA;
		}

		public IntCMAESBuilder setLambda(int lambda) {
			this.lambda = lambda;
			return this;
		}

		public IntCMAESBuilder setMaxEvaluations(int maxEvaluations) {
			this.maxEvaluations = maxEvaluations;
			return this;
		}

		public IntCMAESBuilder setTypicalX(double[] typicalX) {
			this.typicalX = typicalX;
			return this;
		}

		public IntCMAESBuilder setSigma(double sigma) {
			this.sigma = sigma;
			return this;
		}

		public IntCMAESBuilder setMu(double mu) {
			this.mu = mu;
			return this;
		}
		
		public IntCMAESBuilder setCs(double cs) {
			this.cs = cs;
			return this;
		}
		
		public IntCMAESBuilder setCc(double cc) {
			this.cc = cc;
			return this;
		}
		
		public IntCMAESBuilder setCcov(double ccov) {
			this.ccov = ccov;
			return this;
		}
		
		public IntCMAESBuilder setDamps(double damps) {
			this.damps = damps;
			return this;
		}
		
		public IntCMAESBuilder setHC(boolean hc) {
			this.hc = hc;
			return this;
		}

		public IntCovarianceMatrixAdaptationEvolutionStrategy build(long seed) {
			if(mu ==UNDEFINED) {
				mu = 0.5;
			}
			return new IntCovarianceMatrixAdaptationEvolutionStrategy(this,seed);
		}
	}

	@Override
	protected void initProgress() {
		evaluations = 0;
	}

	@Override
	protected void updateProgress() {
		evaluations += lambda;
		updateInternalParameters();
		if(evaluations % 250 == 0) {
			System.out.println("Evalutaions "+evaluations+
					". Current best solution: "+bestSolutionEver.getObjective(0));
		}
	}

	@Override
	protected boolean isStoppingConditionReached() {
		/* Condition number */
		if (StatUtils.min(diagD) <= 0)
			System.out.println("ConditionNumber: smallest eigenvalue smaller or equal zero");
		else if (StatUtils.max(diagD)/StatUtils.min(diagD) > 1e7) {
			System.out.println("ConditionNumber: condition number of the covariance matrix exceeds 1e14");
			int incPopSizeFactor = 2;
			restarts++;
			lambda = (int)Math.ceil(lambda * Math.pow(incPopSizeFactor, restarts));
			System.out.println("Increase lampda, new value: "+lambda);
			this.initializeInternalParameters();
			System.out.println("Re-initializing...");
			population = createInitialPopulation();
		    population = evaluatePopulation(population);
		}
		
		if(evaluations >= maxEvaluations) {
			System.out.println("Reached "+evaluations+" evaluations. Stopping execution.");
			return true;
		}
		
		return false;
	}

	@Override
	protected List<IntegerSolution> createInitialPopulation() {
		List<IntegerSolution> population = new ArrayList<>(lambda);
		for (int i = 0; i < lambda; i++) {
			IntegerSolution newIndividual = getProblem().createSolution();
			population.add(newIndividual);
		}
		return population;
	}

	@Override
	protected List<IntegerSolution> evaluatePopulation(
			List<IntegerSolution> population) {
		for (IntegerSolution solution : population) {
			getProblem().evaluate(solution);
		}
		return population;
	}

	@Override
	protected List<IntegerSolution> selection(
			List<IntegerSolution> population) {
		return population;
	}

	@Override
	protected List<IntegerSolution> reproduction(
			List<IntegerSolution> population) {

		List<IntegerSolution> offspringPopulation = new ArrayList<>(lambda);

		for (int iNk = 0; iNk < lambda; iNk++) {
			offspringPopulation.add(sampleSolution());
		}

		return offspringPopulation;
	}

	@Override
	protected List<IntegerSolution> localImprovement(
			List<IntegerSolution> population) {
		if(hc) {
			// Perform hill climbing with probability pLS
			int rounds = 50;
			
			for (int i = 0; i<population.size(); i++) {
				if(JMetalRandom.getInstance().nextDouble()<pLS) {
//					System.out.println(
//							"Entering HC: Current number of evaluations ("+this.evaluations+")");
					Comparator<IntegerSolution> comparator = new FitnessComparator<>();
					
					HillClimbingMutation operator = new HillClimbingMutation();
					
					LocalSearchOperator<IntegerSolution> localSearch = 
							new BasicLocalSearch<>( rounds, operator, comparator, problem);
					IntegerSolution newSolution = localSearch.execute(population.get(i));
					
					this.evaluations+=localSearch.getEvaluations();
					
					System.out.println(
							"Finished HC: Number of improvements ("+localSearch.getNumberOfImprovements()
							+"). Current number of evaluations ("+this.evaluations+").");
					
//					System.out.println(
//							"Number of evaluations after HC ("+this.evaluations+")");
					population.set(i, newSolution);
				}
			}
		}

		return population;
	}
	
	@Override
	protected List<IntegerSolution> replacement(
			List<IntegerSolution> population,
			List<IntegerSolution> offspringPopulation) {
		return offspringPopulation;
	}

	@Override
	public IntegerSolution getResult() {
		return bestSolutionEver;
	}

	private void initializeInternalParameters() {

		// number of objective variables/problem dimension
		int numberOfVariables = getProblem().getNumberOfVariables();

		// objective variables initial point
		// TODO: Initialize the mean in a better way

		if (typicalX != null) {
			distributionMean = typicalX;
		} else {
			distributionMean = new double[numberOfVariables];
			for (int i = 0; i < numberOfVariables; i++) {
				distributionMean[i] = rand.nextDouble();
			}
		}

		/* Strategy parameter setting: Selection */

		// number of parents/points for recombination
//		mu = (int) Math.floor(lambda / 2);

		// muXone array for weighted recombination
		weights = new double[mu];
		double sum = 0;
		for (int i = 0; i < mu; i++) {
			weights[i] = (Math.log(mu + 1 / 2) - Math.log(i + 1));
			sum += weights[i];
		}
		// normalize recombination weights array
		for (int i = 0; i < mu; i++) {
			weights[i] = weights[i] / sum;
		}

		// variance-effectiveness of sum w_i x_i
		double sum1 = 0;
		double sum2 = 0;
		for (int i = 0; i < mu; i++) {
			sum1 += weights[i];
			sum2 += weights[i] * weights[i];
		}
		muEff = sum1 * sum1 / sum2;

		/* Strategy parameter setting: Adaptation */
		
		
		// time constant for cumulation for C
		if(cumulationC == UNDEFINED) {
			cumulationC = (4 + muEff / numberOfVariables)
					/ (numberOfVariables + 4 + 2 * muEff / numberOfVariables);
		}

		// t-const for cumulation for sigma control
		if(cumulationSigma == UNDEFINED) {
			cumulationSigma = (muEff + 2) / (numberOfVariables + muEff + 5);
		}

		// learning rate for rank-one update of C
		if(c1 == UNDEFINED) {
			c1 = 2 / ((numberOfVariables + 1.3) * (numberOfVariables + 1.3)
					+ muEff);
		}

		// learning rate for rank-mu update
		cmu = Math.min(1 - c1, 2 * (muEff - 2 + 1 / muEff)
				/ ((numberOfVariables + 2) * (numberOfVariables + 2) + muEff));

		// damping for sigma, usually close to 1
		if(dampingSigma == UNDEFINED) {
			dampingSigma = 1
					+ 2 * Math.max(0,
							Math.sqrt((muEff - 1) / (numberOfVariables + 1)) - 1)
					+ cumulationSigma;
		}

		/* Initialize dynamic (internal) strategy parameters and constants */

		// diagonal D defines the scaling
		diagD = new double[numberOfVariables];

		// evolution paths for C and sigma
		pathsC = new double[numberOfVariables];
		pathsSigma = new double[numberOfVariables];

		// b defines the coordinate system
		b = new double[numberOfVariables][numberOfVariables];
		// covariance matrix C
		c = new double[numberOfVariables][numberOfVariables];

		// C^-1/2
		invSqrtC = new double[numberOfVariables][numberOfVariables];

		for (int i = 0; i < numberOfVariables; i++) {
			pathsC[i] = 0;
			pathsSigma[i] = 0;
			diagD[i] = 1;
			for (int j = 0; j < numberOfVariables; j++) {
				b[i][j] = 0;
				invSqrtC[i][j] = 0;
			}
			for (int j = 0; j < i; j++) {
				c[i][j] = 0;
			}
			b[i][i] = 1;
			c[i][i] = diagD[i] * diagD[i];
			invSqrtC[i][i] = 1;
		}

		// track update of b and D
		eigenEval = 0;

		chiN = Math.sqrt(numberOfVariables) * (1 - 1 / (4 * numberOfVariables)
				+ 1 / (21 * numberOfVariables * numberOfVariables));

	}

	private void updateInternalParameters() {

		int numberOfVariables = getProblem().getNumberOfVariables();

		double[] oldDistributionMean = new double[numberOfVariables];
		System.arraycopy(distributionMean, 0, oldDistributionMean, 0,
				numberOfVariables);

		// Sort by fitness and compute weighted mean into distributionMean
		// minimization
		Collections.sort(getPopulation(), comparator);
		storeBest();

		// calculate new distribution mean and BDz~N(0,C)
		updateDistributionMean();

		// Cumulation: Update evolution paths
		int hsig = updateEvolutionPaths(oldDistributionMean);

		// Adapt covariance matrix C
		adaptCovarianceMatrix(oldDistributionMean, hsig);

		// Adapt step size sigma
		double psxps = CMAESUtils.norm(pathsSigma);
		sigma *= Math.exp((cumulationSigma / dampingSigma)
				* (Math.sqrt(psxps) / chiN - 1));

		// Decomposition of C into b*diag(D.^2)*b' (diagonalization)
		decomposeCovarianceMatrix();

	}

	private void updateDistributionMean() {

		int numberOfVariables = getProblem().getNumberOfVariables();

		for (int i = 0; i < numberOfVariables; i++) {
			distributionMean[i] = 0.;
			for (int iNk = 0; iNk < mu; iNk++) {
				double variableValue = (double) getPopulation().get(iNk)
						.getVariableValue(i);
				distributionMean[i] += weights[iNk] * variableValue;
			}
		}

	}

	private int updateEvolutionPaths(double[] oldDistributionMean) {

		int numberOfVariables = getProblem().getNumberOfVariables();

		double[] artmp = new double[numberOfVariables];
		for (int i = 0; i < numberOfVariables; i++) {
			artmp[i] = 0;
			for (int j = 0; j < numberOfVariables; j++) {
				artmp[i] += invSqrtC[i][j]
						* (distributionMean[j] - oldDistributionMean[j])
						/ sigma;
			}
		}
		// cumulation for sigma (pathsSigma)
		for (int i = 0; i < numberOfVariables; i++) {
			pathsSigma[i] = (1. - cumulationSigma) * pathsSigma[i] + Math
					.sqrt(cumulationSigma * (2. - cumulationSigma) * muEff)
					* artmp[i];
		}

		// calculate norm(pathsSigma)^2
		double psxps = CMAESUtils.norm(pathsSigma);

		// cumulation for covariance matrix (pathsC)
		int hsig = 0;
		if ((Math.sqrt(psxps) / Math.sqrt(
				1. - Math.pow(1. - cumulationSigma, 2. * evaluations / lambda))
				/ chiN) < (1.4 + 2. / (numberOfVariables + 1.))) {
			hsig = 1;
		}
		for (int i = 0; i < numberOfVariables; i++) {
			pathsC[i] = (1. - cumulationC) * pathsC[i] + hsig
					* Math.sqrt(cumulationC * (2. - cumulationC) * muEff)
					* (distributionMean[i] - oldDistributionMean[i]) / sigma;
		}

		return hsig;

	}

	private void adaptCovarianceMatrix(double[] oldDistributionMean, int hsig) {

		int numberOfVariables = getProblem().getNumberOfVariables();

		for (int i = 0; i < numberOfVariables; i++) {
			for (int j = 0; j <= i; j++) {
				c[i][j] = (1 - c1 - cmu) * c[i][j]
						+ c1 * (pathsC[i] * pathsC[j] + (1 - hsig) * cumulationC
								* (2. - cumulationC) * c[i][j]);
				for (int k = 0; k < mu; k++) {
					/*
					 * additional rank mu update
					 */
					double valueI = getPopulation().get(k).getVariableValue(i);
					double valueJ = getPopulation().get(k).getVariableValue(j);
					c[i][j] += cmu * weights[k]
							* (valueI - oldDistributionMean[i])
							* (valueJ - oldDistributionMean[j]) / sigma / sigma;
				}
			}
		}

	}

	private void decomposeCovarianceMatrix() {
		int numberOfVariables = getProblem().getNumberOfVariables();

		if (evaluations - eigenEval > lambda / (c1 + cmu) / numberOfVariables
				/ 10) {

			eigenEval = evaluations;

			// enforce symmetry
			for (int i = 0; i < numberOfVariables; i++) {
				for (int j = 0; j <= i; j++) {
					b[i][j] = b[j][i] = c[i][j];
				}
			}

			// eigen decomposition, b==normalized eigenvectors
			double[] offdiag = new double[numberOfVariables];
			CMAESUtils.tred2(numberOfVariables, b, diagD, offdiag);
			CMAESUtils.tql2(numberOfVariables, diagD, offdiag, b);

			checkEigenCorrectness();

			double[][] artmp2 = new double[numberOfVariables][numberOfVariables];
			for (int i = 0; i < numberOfVariables; i++) {
				if (diagD[i] > 0) {
					diagD[i] = Math.sqrt(diagD[i]);
				}
				for (int j = 0; j < numberOfVariables; j++) {
					artmp2[i][j] = b[i][j] * (1 / diagD[j]);
				}
			}
			for (int i = 0; i < numberOfVariables; i++) {
				for (int j = 0; j < numberOfVariables; j++) {
					invSqrtC[i][j] = 0.0;
					for (int k = 0; k < numberOfVariables; k++) {
						invSqrtC[i][j] += artmp2[i][k] * b[j][k];
					}
				}
			}

		}

	}

	private void checkEigenCorrectness() {
		int numberOfVariables = getProblem().getNumberOfVariables();

		if (CMAESUtils.checkEigenSystem(numberOfVariables, c, diagD, b) > 0) {
			evaluations = maxEvaluations;
		}

		for (int i = 0; i < numberOfVariables; i++) {
			// Numerical problem?
			if (diagD[i] < 0) {
				JMetalLogger.logger.severe(
						"CovarianceMatrixAdaptationEvolutionStrategy.updateDistribution:"
								+ " WARNING - an eigenvalue has become negative.");
				evaluations = maxEvaluations;
			}
		}

	}

	private IntegerSolution sampleSolution() {

		IntegerSolution solution = getProblem().createSolution();

		int numberOfVariables = getProblem().getNumberOfVariables();
		double[] artmp = new double[numberOfVariables];
		double sum;

		for (int i = 0; i < numberOfVariables; i++) {
			// TODO: Check the correctness of this random
			// (http://en.wikipedia.org/wiki/CMA-ES)
			artmp[i] = diagD[i] * rand.nextGaussian();
		}
		for (int i = 0; i < numberOfVariables; i++) {
			sum = 0.0;
			for (int j = 0; j < numberOfVariables; j++) {
				sum += b[i][j] * artmp[j];
			}

			int value = (int) (distributionMean[i] + sigma * sum);
			if (value > ((AbstractIntegerProblem) getProblem())
					.getUpperBound(i)) {
				value = ((AbstractIntegerProblem) getProblem())
						.getUpperBound(i);
			} else if (value < ((AbstractIntegerProblem) getProblem())
					.getLowerBound(i)) {
				value = ((AbstractIntegerProblem) getProblem())
						.getLowerBound(i);
			}

			solution.setVariableValue(i, value);
		}

		return solution;
	}

	private void storeBest() {
		if ((bestSolutionEver == null) || (bestSolutionEver
				.getObjective(0) > getPopulation().get(0).getObjective(0))) {
			bestSolutionEver = getPopulation().get(0);
		}
	}

	@Override
	public String getName() {
		return "CMAES";
	}

	@Override
	public String getDescription() {
		return "Covariance Matrix Adaptation Evolution Strategy";
	}

}