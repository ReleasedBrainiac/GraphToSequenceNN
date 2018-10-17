from DatasetHandler.ContentSupport import isList

class EvaluationHelpers:

    def CalculateMeanValue(self, sentences_length):
        """
        This function calculate the mean over all values in a list.
            :param sentences_length: lengths of all sentences
        """
        if(isList(sentences_length)):
            sent_summ = 0

            for index, _ in enumerate(sentences_length):
                sent_summ += sentences_length[index]

            mw = int(round(sent_summ / len(sentences_length)))
            return mw
        else:
            print('WRONG INPUT FOR [CalculateMeanValue]')
            return None