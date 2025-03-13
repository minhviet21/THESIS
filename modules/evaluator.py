class Evaluator:
    def evaluate(self, grouth_truths, predictions):
        true = 0
        false = 0
        for grouth_truth, prediction in zip(grouth_truths, predictions):
            if bool(set(grouth_truth) & set(prediction)):
                true += 1
            else:
                false += 1
        return true / (true + false)

            
        