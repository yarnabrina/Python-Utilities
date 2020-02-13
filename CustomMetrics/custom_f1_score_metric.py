"""
Custom F1 Score Metric
"""

import tensorflow


class MacroF1ScoreMetric(tensorflow.keras.metrics.Metric):
    """
    Custom metric to find pooled F1 score based on macro averaging over individual class scores

    Parameters
    ----------
    number_of_categories : int
        number of possible classes of the tickets
    name : str, optional
        name of the metric, by default "f1_macro_score"
    dtype : tensorflow.DType, optional
        type of the metric, by default tensorflow.float64
    """

    def __init__(self,
                 number_of_categories: int,
                 name: str = "f1_macro_score",
                 dtype: tensorflow.DType = tensorflow.float64) -> None:
        super(MacroF1ScoreMetric, self).__init__(name=name, dtype=dtype)

        self.number_of_categories = number_of_categories

        self.true_positives, self.false_positives, self.false_negatives = map(
            lambda custom_name: self.add_weight(name=custom_name,
                                                shape=(self.
                                                       number_of_categories,),
                                                initializer="zeros",
                                                dtype=self.dtype),
            ("true_positives", "false_positives", "false_negatives"))

    def update_state(self,
                     y_true: tensorflow.Tensor,
                     y_pred: tensorflow.Tensor,
                     sample_weight: tensorflow.Tensor = None) -> None:
        """
        Updates the variables keeeping track of progress during an epoch

        Parameters
        ----------
        y_true : tensorflow.Tensor
            one hot encoded true classifications
        y_pred : tensorflow.Tensor
            predicted probabilities
        sample_weight : tensorflow.Tensor, optional
            weight for each observation, by default None
        """
        # TODO add support for sample weights
        del sample_weight  # unused

        y_true = tensorflow.math.argmax(y_true, axis=1)
        y_pred = tensorflow.math.argmax(y_pred, axis=1)

        # row: true, column: predictions
        confusion_matrix = tensorflow.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.number_of_categories,
            dtype=self.dtype)

        diagonals = tensorflow.linalg.diag_part(confusion_matrix)
        column_sums = tensorflow.math.reduce_sum(confusion_matrix, axis=0)
        row_sums = tensorflow.math.reduce_sum(confusion_matrix, axis=1)

        # belongs to positive class, and predicted so
        self.true_positives.assign_add(diagonals)
        # does not belong to positive class, but predicted so
        self.false_positives.assign_add(column_sums - diagonals)
        # belongs to positive class, but not predicted so
        self.false_negatives.assign_add(row_sums - diagonals)

    def result(self) -> tensorflow.float64:
        """
        Calculates F1 scores for each class and returns the pooled F1 score after macro-averaging

        Returns:
            tensorflow.float64
                -- pooled F1 score
        """
        precisions = tensorflow.math.divide_no_nan(
            self.true_positives,
            tensorflow.math.add(self.true_positives, self.false_positives))
        recalls = tensorflow.math.divide_no_nan(
            self.true_positives,
            tensorflow.math.add(self.true_positives, self.false_negatives))

        # TODO add support for weighted harmonic mean of precision and recall
        f1_scores = tensorflow.math.divide_no_nan(
            tensorflow.math.scalar_mul(
                2, tensorflow.math.multiply(precisions, recalls)),
            tensorflow.math.add(precisions, recalls))

        # TODO add support for weighted average of f1 scores for each class
        macro_f1_score = tensorflow.math.reduce_mean(f1_scores)

        return macro_f1_score

    def reset_states(self) -> None:
        """
        Resets the variables at the end of an epoch
        """
        tensorflow.keras.backend.batch_set_value([
            (variable, tensorflow.zeros((self.number_of_categories,)))
            for variable in self.variables
        ])