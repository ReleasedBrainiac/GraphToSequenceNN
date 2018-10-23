from DatasetHandler.ContentSupport import isList

class EvaluationHelpers:

    def CalculateMeanValue(self, sentences_length):
        """
        This function calculate the mean over all values in a list.
            :param sentences_length: lengths of all sentences
        """
        try:
            sent_summ = 0

            for index, _ in enumerate(sentences_length):
                sent_summ += sentences_length[index]

            mw = int(round(sent_summ / len(sentences_length)))
            return mw
        except ValueError:
            print("ERR: Missing or wrong value passed to [CalculateMeanValue].")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        