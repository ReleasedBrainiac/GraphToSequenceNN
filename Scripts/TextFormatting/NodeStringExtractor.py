from ContentSupport import isInStr, isNotInStr, isNotNone, isStr

def CheckEqualOpenEnclosingParenthesis(content, open_par, closing_par):
    count_open = content.count(open_par)
    count_close = content.count(closing_par)

    if (count_open == count_close):
        return True
    else:
        return False

def GetEnclosedContent(content, open_par, closing_par):
    pos_open = content.index(open_par)
    pos_close = content.rfind(closing_par)

    return content[pos_open+1:pos_close-1]

def GetParenthesisIndexPairs(content, open_par, closing_par):
    open_indices = []
    pairs = []

    for index in range(len(content)):
        char = content[index]
        if char is open_par:
            open_indices.append(index)

        if char is closing_par:
            pairs.append([open_indices.pop(), index, len(open_indices)])

    return pairs

def GetNodes(content, remain, index, pairs, open_par, closing_par):
    pair = pairs[index]
    raw = content[pair[0]:pair[1]+1]
    result = GetEnclosedContent(raw, open_par, closing_par)

    if open_par in result and closing_par in result:
        print('Outer: ',result)
        GetNodes(content, remain, index+1, pairs, open_par, closing_par)
    else:
        print('Most_Inner: ',result)
        pairs.remove(index)
        GetNodes(content, remain, 0, pairs, open_par, closing_par)

        























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


print(CheckEqualOpenEnclosingParenthesis(content_example, '(', ')'))
#print(GetEnclosedContent(content_example, open_par, closing_par))
pairs = GetParenthesisIndexPairs(content_example, open_par, closing_par)
print(pairs)
print(content_example)

#for pair in pairs:
#    print('[', content_example[pair[0]:pair[1]+1] ,'] in depth [',pair[2],']') 

print(GetNodes(content_example, content_example, 0, pairs, open_par, closing_par))