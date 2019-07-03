import sys, re
from Plotter.ReloadHistory import HistoryLoader

class TestClassPlotter():

    DATASET:str = "graph2seq_model_Der Kleine Prinz AMR_DT_20190701 21_33_39/Der Kleine Prinz AMR_Log.log"

    def Execute(self):
        try:
            plotter = HistoryLoader(self.DATASET)
            plotter.CollectContextFromLog()
            plotter.CollectingRespone()
        except Exception as ex:
            template = "An exception of type {0} occurred in [TestClassPlotter.Main]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            sys.exit(1)

if __name__ == "__main__":
    TestClassPlotter().Execute()