from __future__ import division

'''
This class provide a simple number to qualified name parser.
It is based on the following resources:
    => https://codereview.stackexchange.com/questions/173971/converting-number-to-words
    => https://stackoverflow.com/questions/1906717/splitting-integer-in-python 
'''
class NumWordParser:

    ones_tens = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    ten_potencies = ["Ten","Twenty", "Thirty", "Fourty", "Fifty", "Sixty", "Seventy","Eighty", "Ninety"]
    hundreds = "Hundred"
    thousands = "Thousand"
    millions = "Million"
    billions = "Billion"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def MergeForamted(self, first_word, second_word):
        if (first_word != None) and (second_word != None):
            if (len(first_word) < 1):
                return second_word
            elif (len(second_word) < 1):
                return first_word
            else:
                return first_word + ' ' + second_word
        else:
            return None

    def GetDigitsByBase(self, number, base=1000):
        assert number >= 0
        if number == 0:
            return [0]
        l = []
        while number > 0:
            l.insert(0, number % base)
            number = number // base
        return l

    def GetReversedDigitsByBase(self, number, base=1000):
        assert number >= 0
        if number == 0:
            return [0]
        l = []
        while number > 0:
            l.append(number % base)
            number = number // base
        return l

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def GetIntervalZroToTwenty(self, number):
        digits = self.GetDigitsByBase(number, 10)
        if digits != None and len(digits) > 0 and len(digits) < 3 and number < 20:
            return self.ones_tens[number]
        else:
            return None

    def GetTenPontecies(self, number):
        digits = self.GetDigitsByBase(number, 10)
        if digits != None and len(digits) > 1 and len(digits) < 3 and number > 19:
            return self.MergeForamted(self.ten_potencies[digits[0]-1], self.GetIntervalZroToTwenty(digits[1]))
        else:
            return None

    def GetHundreds(self, number):
        digits = self.GetDigitsByBase(number, 10)
        one_ten_digit = ((digits[1] * 10) + digits[2])

        if digits != None and len(digits) > 2 and len(digits) < 4:
            hundred = self.MergeForamted(self.GetIntervalZroToTwenty(digits[0]), self.hundreds)

            if (one_ten_digit < 19):
                return self.MergeForamted(hundred, self.GetIntervalZroToTwenty(one_ten_digit))
            else:
                return self.MergeForamted(hundred, self.GetTenPontecies(one_ten_digit))
        else:
            return None
    
    def GetThounsands(self, number):
        digits = self.GetDigitsByBase(number, 10)
        hundreds = self.GetDigitsByBase(number, 1000)
        if hundreds != None and digits != None and len(digits) > 3 and len(digits) < 7:
            thousand = self.MergeForamted(self.GetSelectCorrectWordUntilHundred(hundreds[0]) , self.thousands)
            return self.MergeForamted(thousand, self.GetSelectCorrectWordUntilHundred(hundreds[1]))
        else:
            return None

    def GetMillions(self, number):
        digits = self.GetDigitsByBase(number, 10)
        hundreds = self.GetDigitsByBase(number, 1000)
        if hundreds != None and digits != None and len(digits) > 6 and len(digits) < 10:
            million = self.MergeForamted(self.GetSelectCorrectWordUntilHundred(hundreds[0]) , self.millions) 
            rest = (hundreds[1] * 1000) + hundreds[2]
            if(rest > 0):
                return self.MergeForamted(million, self.GetThounsands(rest))
            else: 
                return million
        else:
            return None

    def GetBillions(self, number):
        digits = self.GetDigitsByBase(number, 10)
        hundreds = self.GetDigitsByBase(number, 1000)
        if hundreds != None and digits != None and len(digits) > 9 and len(digits) < 13:
            billion = self.MergeForamted(self.GetSelectCorrectWordUntilHundred(hundreds[0]) , self.billions) 
            rest = (hundreds[1] * 1000000) + (hundreds[2] * 1000) + hundreds[3]
            if(rest > 0):
                return self.MergeForamted(billion, self.GetMillions(rest))
            else: 
                return billion
        else:
            return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def GetSelectCorrectWordUntilHundred(self, number):
        digits = self.GetDigitsByBase(number, 10)
        if(digits != None and len(digits) > 0 and len(digits) < 4):
            if len(digits) < 3 and number < 20:
                return self.GetIntervalZroToTwenty(number)
            elif(len(digits) > 1 and len(digits) < 3 and number > 19):
                return self.MergeForamted(self.ten_potencies[digits[0]-1], self.GetIntervalZroToTwenty(digits[1]))
            else:
                return self.GetHundreds(number)
        else:
            return None

    def BillionNumberToEnglishWord(self, number):
        digits = self.GetDigitsByBase(number, 1000)

        if (len(digits) == 1):
            if(digits[0] == 0):
                return 'Zero'
            else:
                return self.GetSelectCorrectWordUntilHundred(number)
        elif (len(digits) == 2):
            return self.GetThounsands(number)
        elif (len(digits) == 3):
            return self.GetMillions(number)
        elif (len(digits) == 4):
            return self.GetBillions(number)
        else:
            print('Input [',number,'] was negativ or greater then billion')
            return None