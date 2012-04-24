import java.util.Random;

import com.evolvingstuff.EmbeddedReberGrammar;
import com.evolvingstuff.LSTM;

public class Test {
	public static void main(String[] args) throws Exception {
		System.out.println("Test of Long Short Term Memory on Embedded Reber Grammar task");
		Random r = new Random(1234);
		EmbeddedReberGrammar.deterministic_evaluation = false;
		EmbeddedReberGrammar.reset_at_begining = true;
		EmbeddedReberGrammar evaluator = new EmbeddedReberGrammar(r);
		int cell_blocks = 15;
		LSTM agent = new LSTM(r, evaluator.GetObservationDimension(), evaluator.GetActionDimension(), cell_blocks);
		int training_epoches = 1000;
		for (int t = 0; t < training_epoches; t++)
		{
			evaluator.SetValidationMode(false);
			double fit = evaluator.EvaluateFitnessSupervised(agent);
			evaluator.SetValidationMode(true);
			double validation = evaluator.EvaluateFitnessSupervised(agent);
			System.out.println("\t["+t+"]:\t" + (1-fit) + "\t" + (1-validation));
		}
		System.out.println("done.");
	}

}
