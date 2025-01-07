import openpyxl
import pandas as pd

PATH        = "C:\\Desktop\\"
FILE_NAME   = "Tides.xlsx"
df          = pd.read_excel(PATH + FILE_NAME, sheet_name='Sheet1')
df.to_excel(PATH + "NewFile.xlsx", sheet_name='Sheet1')
print (df)