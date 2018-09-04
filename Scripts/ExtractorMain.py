# - *- coding: utf-8*-
from TextFormatting.NodeStringExtractor import CheckOpenEnclosing, BuildCleanDefinedAMR
open_par = '('
closing_par = ')'

content_example = """(c / cause-01
      :ARG1 (l / live-01
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


print('Open-Close valid? [',CheckOpenEnclosing(content_example, '(', ')'),']')
print(content_example)
print(BuildCleanDefinedAMR(content_example, open_par, closing_par))