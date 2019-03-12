from __future__ import division
from DatasetHandler.ContentSupport import toInt

class NumWordParser():
    """
    Attention UNUSED!
    This class provide a simple number to qualified word representation parser.
    It is based on the following resources:
        => https://codereview.stackexchange.com/questions/173971/converting-number-to-words
        => https://stackoverflow.com/questions/1906717/splitting-integer-in-python 

    """
    until_nineteen = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    ten_potencies = ["Ten","Twenty", "Thirty", "Fourty", "Fifty", "Sixty", "Seventy","Eighty", "Ninety"]
    hundreds = "Hundred"
    thousands = "Thousand"
    millions = "Million"
    billions = "Billion"

    def __init__(self, number):
        """
        This constructor collect the given number.
        It also convert the number and provide the result at the value "result"
            :param number: given number as string or int or float
        """   
        try:
            self.in_content = toInt(number)
            self.result = self.Convert(self.in_content)
        except ValueError:
            print("No valid number passed to [NumWordParser.Constructor]. Try again...")

    def MergeForamted(self, first_word:str, second_word:str):
        """
        This function build a merge string of two given words.
            :param first_word:str: first number string 
            :param second_word:str: second number string
        """   
        try:
            if (first_word != None) and (second_word != None):
                if (len(first_word) < 1):
                    return second_word
                elif (len(second_word) < 1):
                    return first_word
                else:
                    return first_word + ' ' + second_word
            else:
                return None
        except ValueError:
            print("Oops!  That was no valid number to [NumWordParser.MergeForamted].  Try again...")

    def GetDigitsByBase(self, in_value:int =-1, base:int =1000):
        """
        This function split a number into a list of numbers depending on the given base value.
            :param in_value:int: the number
            :param base:int: the base a 10th potency ~> [1000] => numbers are splitted into [0;999] arrays
        """   
        try:
            if (in_value >= 0):
                number = int(in_value)
            else:
                number = self.in_content
            assert number >= 0
            if number == 0:
                return [0]
            l = []
            while number > 0:
                l.insert(0, number % base)
                number = number // base
            return l
        except ValueError:
            print("No valid number passed to [NumWordParser.GetDigitsByBase]. Try again...")

    def GetDigitsByBaseReversed(self, in_value:int =-1, base:int =1000):
        """
        This function split a number into a reversed list of numbers depending on the given base value.
            :param in_value:int: the number
            :param base:int: the base a 10th potency ~> [1000] => numbers are splitted into [0;999] arrays
        """   
        try:
            if (in_value >= 0):
                number = int(in_value)
            else:
                number = self.in_content
            assert number >= 0
            if number == 0:
                return [0]
            l = []
            while number > 0:
                l.append(number % base)
                number = number // base
            return l
        except ValueError:
            print("No valid number passed to [NumWordParser.GetDigitsByBaseReversed]. Try again...")

    def GetUntilNineteenRepresentation(self, number:int):
        """
        This function return a representation string for all values [0; 19]
            :param number:int: given number
        """
        try:
            digits = self.GetDigitsByBase(int(number), 10)
            if digits != None and len(digits) > 0 and len(digits) < 3 and number < 20:
                return self.until_nineteen[number]
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetUntilNineteenRepresentation]. Try again...")

    def GetUntilNinetyNineRepresentation(self, number:int):
        """
        This function return a representation string for all values [0; 99]
            :param number:int: given number
        """   
        try:
            digits = self.GetDigitsByBase(number, 10)
            if digits != None and len(digits) > 1 and len(digits) < 3 and number > 19:
                return self.MergeForamted(self.ten_potencies[digits[0]-1], self.GetUntilNineteenRepresentation(digits[1]))
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetUntilNinetyNineRepresentation]. Try again...")

    def GetHundredsRepresentation(self, number:int):
        """
        This function return a representation string for all values [0; 999]
            :param number:int: given number
        """ 
        try:
            digits = self.GetDigitsByBase(int(number), 10)
            one_ten_digit = ((digits[1] * 10) + digits[2])

            if digits != None and len(digits) > 2 and len(digits) < 4:
                hundred = self.MergeForamted(self.GetUntilNineteenRepresentation(digits[0]), self.hundreds)

                if (one_ten_digit < 19):
                    return self.MergeForamted(hundred, self.GetUntilNineteenRepresentation(one_ten_digit))
                else:
                    return self.MergeForamted(hundred, self.GetUntilNinetyNineRepresentation(one_ten_digit))
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetHundredsRepresentation]. Try again...")
    
    def GetThounsandsRepresentation(self, number:int):
        """
        This function return a representation string for all values [0; 999.999]
            :param number:int: given number
        """ 
        try:
            digits = self.GetDigitsByBase(int(number), 10)
            hundreds = self.GetDigitsByBase(number, 1000)
            if hundreds != None and digits != None and len(digits) > 3 and len(digits) < 7:
                thousand = self.MergeForamted(self.GetSelectCorrectWordUntilHundred(hundreds[0]) , self.thousands)
                return self.MergeForamted(thousand, self.GetSelectCorrectWordUntilHundred(hundreds[1]))
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetThounsandsRepresentation]. Try again...")

    def GetMillionsRepresentation(self, number:int):
        """
        This function return a representation string for all values [0; 999.999.999]
            :param number:int: given number
        """ 
        try:
            digits = self.GetDigitsByBase(int(number), 10)
            hundreds = self.GetDigitsByBase(number, 1000)
            if hundreds != None and digits != None and len(digits) > 6 and len(digits) < 10:
                million = self.MergeForamted(self.GetSelectCorrectWordUntilHundred(hundreds[0]) , self.millions) 
                rest = (hundreds[1] * 1000) + hundreds[2]
                if(rest > 0):
                    return self.MergeForamted(million, self.GetThounsandsRepresentation(rest))
                else: 
                    return million
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetMillionsRepresentation]. Try again...")

    def GetBillionsRepresentation(self, number:int):
        """
        This function return a representation string for all values [0; 999.999.999.999]
            :param number:int: given number
        """ 
        try:
            digits = self.GetDigitsByBase(int(number), 10)
            hundreds = self.GetDigitsByBase(number, 1000)
            if hundreds != None and digits != None and len(digits) > 9 and len(digits) < 13:
                billion = self.MergeForamted(self.GetSelectCorrectWordUntilHundred(hundreds[0]) , self.billions) 
                rest = (hundreds[1] * 1000000) + (hundreds[2] * 1000) + hundreds[3]
                if(rest > 0):
                    return self.MergeForamted(billion, self.GetMillionsRepresentation(rest))
                else: 
                    return billion
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetBillionsRepresentation]. Try again...")

    def GetSelectCorrectWordUntilHundred(self, number:int):
        """
        This function returnss the word representation for values in [0; 100].
            :param number:int: value out of [0; 100]
        """   
        try:
            digits = self.GetDigitsByBase(int(number), 10)
            if(digits != None and len(digits) > 0 and len(digits) < 4):
                if len(digits) < 3 and number < 20:
                    return self.GetUntilNineteenRepresentation(number)
                elif(len(digits) > 1 and len(digits) < 3 and number > 19):
                    return self.MergeForamted(self.ten_potencies[digits[0]-1], self.GetUntilNineteenRepresentation(digits[1]))
                else:
                    return self.GetHundredsRepresentation(number)
            else:
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.GetSelectCorrectWordUntilHundred]. Try again...")

    def Convert(self, number:int):
        """
        This function converts the given number if its at least inside [0; 100.000.000.000] to its word representation.
            :param number:int: given integer 
        """   
        try:
            digits = self.GetDigitsByBase(int(number), 1000)

            if (len(digits) == 1):
                if(digits[0] == 0):
                    return 'Zero'
                else:
                    return self.GetSelectCorrectWordUntilHundred(number)
            elif (len(digits) == 2):
                return self.GetThounsandsRepresentation(number)
            elif (len(digits) == 3):
                return self.GetMillionsRepresentation(number)
            elif (len(digits) == 4):
                return self.GetBillionsRepresentation(number)
            else:
                print('Input [',number,'] was negativ or greater then billion')
                return None
        except ValueError:
            print("No valid number passed to [NumWordParser.Convert]. Try again...")

        