package org.uma.jmetal.operator.impl.mutation;

import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

public class RealHillClimbingMutation  implements MutationOperator<DoubleSolution> {

	/**
	 * Generated.
	 */
	private static final long serialVersionUID = -7139532838014201724L;

	@Override
	public DoubleSolution execute(DoubleSolution solution) {
		
		boolean success = false;
		
		double step = 0.001;
		
//		System.out.println("Original chromosome: "+solution.getVariables().toString());
		
		while(!success) {
			int chromosome = 
					JMetalRandom.getInstance().nextInt(0, solution.getNumberOfVariables()-1);
			
			if(JMetalRandom.getInstance().nextDouble()>0.5) {
				// Increase value
				if(solution.getVariables().get(chromosome)+step<solution.getUpperBound(chromosome)) {
					double newValue = solution.getVariables().get(chromosome)+step;
					solution.setVariableValue(chromosome, newValue);
					
					success= true;
				}
			} else {
				// Reduce value
				if(solution.getVariables().get(chromosome)-step>solution.getLowerBound(chromosome)) {
					double newValue = solution.getVariables().get(chromosome)-step;
					solution.setVariableValue(chromosome, newValue);
					
					success= true;
				}
			}
		
		}
		
//		System.out.println("Modified chromosome: "+solution.getVariables().toString());
		
		return solution;
	}
}