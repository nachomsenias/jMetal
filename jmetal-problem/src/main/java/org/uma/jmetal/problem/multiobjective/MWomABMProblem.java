package org.uma.jmetal.problem.multiobjective;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.uma.jmetal.problem.impl.AbstractIntegerDoubleProblem;
import org.uma.jmetal.solution.IntegerDoubleSolution;
import org.uma.jmetal.solution.impl.DefaultIntegerDoubleSolution;

import com.zioabm.beans.CalibrationConfig;
import com.zioabm.beans.SimulationConfig;

import calibration.CalibrationParameter;
import calibration.CalibrationParametersManager;
import calibration.fitness.history.HistoryManager;
import calibration.fitness.history.ScoreBean.ScoreWrapper;
import model.ModelDefinition;
import model.ModelManager;
import model.ModelRunner;
import util.StringBean;
import util.exception.calibration.CalibrationException;
import util.statistics.MonteCarloStatistics;

public class MWomABMProblem extends AbstractIntegerDoubleProblem<IntegerDoubleSolution> {

	protected int mcIterations;
	protected ModelDefinition md;
	protected StringBean[] parameters;

	protected CalibrationParametersManager paramManager;
	protected HistoryManager manager;
	private ModelManager modelManager;

	/**
	 * Generated id.
	 */
	private static final long serialVersionUID = 2656241940318033406L;

	public MWomABMProblem(CalibrationConfig cc) throws CalibrationException {

		parameters = cc.getCalibrationModelParameters();

		int numberOfVariables = parameters.length;
		setNumberOfVariables(numberOfVariables + 1);
		setNumberOfIntegerVariables(1);
		setNumberOfDoubleVariables(numberOfVariables);

		setNumberOfObjectives(2);
		setName("MWomABMProblem");

		// Set problem
		SimulationConfig simConfig = cc.getSimConfig();
		mcIterations = simConfig.getnMC();
		md = simConfig.getModelDefinition();
		paramManager = new CalibrationParametersManager(parameters, md, false);
		manager = cc.getHistoryManager();
		modelManager = new ModelManager(md, paramManager.getInvolvedDrivers());

		List<Number> lowerLimit = new ArrayList<Number>(numberOfVariables + 1);
		List<Number> upperLimit = new ArrayList<Number>(numberOfVariables + 1);

		// M parameter is treated separately
		// Min M = 2;
		// MAX M = 8;
		lowerLimit.add(new Integer(2));
		upperLimit.add(new Integer(8));

		for (int i = 0; i < numberOfVariables; i++) {

			double[] paramDef = paramManager
					.parseParameterValue(parameters[i].getValue());
			// Min
			lowerLimit.add(paramDef[0]);
			// Max
			upperLimit.add(paramDef[1]);
		}

		setLowerLimit(lowerLimit);
		setUpperLimit(upperLimit);
	}

	protected void invoke(CalibrationParameter param, double value)
			throws IllegalAccessException, IllegalArgumentException,
			InvocationTargetException {

		int[] indexes = param.indexes;
		Method setter = param.setterMethod;

		if (indexes == null) {
			setter.invoke(modelManager, value);
		} else {
			switch (indexes.length) {
			case 0:
				throw new IllegalStateException(
						"Undefined indexes for parameter: " + param.parameterName);
			case 1:
				setter.invoke(modelManager, indexes[0], value);
				break;
			case 2:
				setter.invoke(modelManager, indexes[0], indexes[1], value);
				break;
			case 3:
				setter.invoke(modelManager, indexes[0], indexes[1], indexes[2], value);
				break;
			case 4:
				setter.invoke(modelManager, indexes[0], indexes[1], indexes[2],
						indexes[3], value);
				break;
			default:
				throw new UnsupportedOperationException(
						"Cannot invoke setter with " + indexes.length + " indexes");
			}
		}
	}

	public ModelDefinition getMD() {
		return md;
	}

	@Override
	public void evaluate(IntegerDoubleSolution solution) {
		// Set values into model definition instance.
		List<CalibrationParameter> modelParams = paramManager.getParameters();

		// M parameter is treated separately
		modelManager.setMValue(0, solution.getVariableValue(0).intValue());

		int variables = parameters.length;
		for (int v = 0; v < variables; v++) {
			try {
				Number value = solution.getVariableValue(v + 1);

				invoke(modelParams.get(v), value.doubleValue());
			} catch (IllegalAccessException e) {
				e.printStackTrace();
			} catch (IllegalArgumentException e) {
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				e.printStackTrace();
			}
		}

		MonteCarloStatistics mcStats = ModelRunner.simulateModel(md, mcIterations, false,
				manager.getStatsBean());

		// Store simulation values
		ScoreWrapper wrapper = manager.computeTrainingScore(mcStats);
		solution.setObjective(0, wrapper.awarenessScore.getScore());
		solution.setObjective(1, wrapper.womVolumeScore.getScore());
	}

	@Override
	public IntegerDoubleSolution createSolution() {
		return new DefaultIntegerDoubleSolution(this);
	}

	public static DefaultIntegerDoubleSolution cloneSolution(
			AbstractIntegerDoubleProblem<IntegerDoubleSolution> problem,
			IntegerDoubleSolution sol) {
		DefaultIntegerDoubleSolution newSolution = new DefaultIntegerDoubleSolution(
				problem);

		for (int v = 0; v < sol.getNumberOfVariables(); v++) {
			newSolution.setVariableValue(v, sol.getVariableValue(v));
		}

		for (int o = 0; o < sol.getNumberOfObjectives(); o++) {
			newSolution.setObjective(o, sol.getObjective(o));
		}

		return newSolution;
	}
}
