class EvaluationHelpers():
    """
    This class provides arithmetic operations like mean calculation over a list of numerical informations.
    """

    def CalculateMeanValue(self, sentences_lengths:list):
        """
        This function calculate the mean over all values in a list.
            :param sentences_lengths:list: lengths of all sentences
        """
        try:
            sent_summ = 0

            for index, _ in enumerate(sentences_lengths):
                sent_summ += sentences_lengths[index]

            mw = int(round(sent_summ / len(sentences_lengths)))
            return mw
        except Exception as ex:
            template = "An exception of type {0} occurred in [TextEvaluation.CalculateMeanValue]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        