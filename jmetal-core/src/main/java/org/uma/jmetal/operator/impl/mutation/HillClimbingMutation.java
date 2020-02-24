package org.uma.jmetal.operator.impl.mutation;

import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.solution.IntegerSolution;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

public class HillClimbingMutation implements MutationOperator<IntegerSolution> {

	/**
	 * Generated.
	 */
	private static final long serialVersionUID = 521953305308774283L;

	@Override
	public IntegerSolution execute(IntegerSolution solution) {
		
		boolean success = false;
		
//		System.out.println("Original chromosome: "+solution.getVariables().toString());
		
		while(!success) {
			int chromosome = 
					JMetalRandom.getInstance().nextInt(0, solution.getNumberOfVariables()-1);
			
			if(JMetalRandom.getInstance().nextDouble()>0.5) {
				// Increase value
				if(solution.getVariables().get(chromosome)<solution.getUpperBound(chromosome)) {
					int newValue = solution.getVariables().get(chromosome)+1;
					solution.setVariableValue(chromosome, newValue);
					
					success= true;
				}
			} else {
				// Reduce value
				if(solution.getVariables().get(chromosome)>solution.getLowerBound(chromosome)) {
					int newValue = solution.getVariables().get(chromosome)-1;
					solution.setVariableValue(chromosome, newValue);
					
					success= true;
				}
			}
		
		}
		
//		System.out.println("Modified chromosome: "+solution.getVariables().toString());
		
		return solution;
	}
}
