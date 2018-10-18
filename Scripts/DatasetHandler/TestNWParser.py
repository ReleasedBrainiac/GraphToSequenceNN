from NumberToWordParser import NumWordParser

parser = NumWordParser()

test1 = int(104382426112)
process1 = parser.BillionNumberToEnglishWord(test1)
result1 = "One Hundred Four Billion Three Hundred Eighty Two Million Four Hundred Twenty Six Thousand One HUndred Twelve"

test2 = int(1253456321)
process2 = parser.BillionNumberToEnglishWord(test2)
result2 = "One Billion Two Hundred Fifty Three Million Four Hundred Fifty Six Thousand Three Hundred Twenty One"

print('O: ',test1,'| R: ',process1,']')
print(process1 != result1)

print('O: ',test2,'| R: ',process2,']')
print(process2 != result2)

print('O: ',0,'| R: ',parser.BillionNumberToEnglishWord(0),']')
print('O: ',4,'| R: ',parser.BillionNumberToEnglishWord(4),']')
print('O: ',10,'| R: ',parser.BillionNumberToEnglishWord(10),']')
print('O: ',32,'| R: ',parser.BillionNumberToEnglishWord(32),']')
print('O: ',110,'| R: ',parser.BillionNumberToEnglishWord(110),']')
print('O: ',324,'| R: ',parser.BillionNumberToEnglishWord(324),']')


print('O: ',1004,'| R: ',parser.BillionNumberToEnglishWord(1004),']')
print('O: ',1324,'| R: ',parser.BillionNumberToEnglishWord(1324),']')
print('O: ',10000,'| R: ',parser.BillionNumberToEnglishWord(10000),']')
print('O: ',13124,'| R: ',parser.BillionNumberToEnglishWord(13124),']')
print('O: ',100000,'| R: ',parser.BillionNumberToEnglishWord(100000),']')
print('O: ',132114,'| R: ',parser.BillionNumberToEnglishWord(132114),']')


print('O: ',1000000,'| R: ',parser.BillionNumberToEnglishWord(1000000),']')
print('O: ',1321142,'| R: ',parser.BillionNumberToEnglishWord(1321142),']')
print('O: ',10000000,'| R: ',parser.BillionNumberToEnglishWord(10000000),']')
print('O: ',13214514,'| R: ',parser.BillionNumberToEnglishWord(13214514),']')
print('O: ',100000000,'| R: ',parser.BillionNumberToEnglishWord(100000000),']')
print('O: ',132113204,'| R: ',parser.BillionNumberToEnglishWord(132113204),']')


print('O: ',1000000000,'| R: ',parser.BillionNumberToEnglishWord(1000000000),']')
print('O: ',1321132904,'| R: ',parser.BillionNumberToEnglishWord(1321132904),']')
print('O: ',10000000000,'| R: ',parser.BillionNumberToEnglishWord(10000000000),']')
print('O: ',13211320434,'| R: ',parser.BillionNumberToEnglishWord(13211320434),']')
print('O: ',100000000000,'| R: ',parser.BillionNumberToEnglishWord(100000000000),']')
print('O: ',132113203404,'| R: ',parser.BillionNumberToEnglishWord(132113203404),']')