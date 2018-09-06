# - *- coding: utf-8*-
from GraphHandler.NodeStringExtractor import CheckOpenEnclosing, GenerateCleanAMR
open_par = '('
closing_par = ')'

example_1 = """(c / cause-01
      :ARG1 (l / live-01 :polarity -
            :ARG0 (i / i
                  :ARG0-of (t3 / talk-01 :polarity -
                        :ARG2 (a5 / anyone)
                        :ARG1-of (r / real-04)))
            :ARG1 (l2 / life
                  :poss i)
            :manner (a / alone)
            :duration (u / until
                  :op1 (h / have-06
                        :ARG0 i
                        :ARG1 (a3 / accident
                              :mod (p / plane))
                        :location (d / desert :wiki "Sahara" :name (n / name :op1 "Desert" :op2 "of" :op3 "Sahara"))
                        :time (b / before
                              :op1 (n2 / now)
                              :quant (t2 / temporal-quantity :quant 6
                                    :unit (y / year)))))))"""


example_2 = """(c / chapter
  :mod 1)
"""

example_3 = """(y2 / yes)"""

print(example_1)
result = GenerateCleanAMR(example_1, open_par, closing_par)
print(result)
print('Open-Close valid? [',CheckOpenEnclosing(result, '(', ')'),']')
print('\n#####################################\n')

print(example_2)
result = GenerateCleanAMR(example_2, open_par, closing_par)
print(result)
print('Open-Close valid? [',CheckOpenEnclosing(result, '(', ')'),']')
print('\n#####################################\n')

print(example_3)
result = GenerateCleanAMR(example_3, open_par, closing_par)
print(result)
print('Open-Close valid? [',CheckOpenEnclosing(result, '(', ')'),']')