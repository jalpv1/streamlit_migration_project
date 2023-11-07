
import pandas as pd

read_file = pd.read_excel ('data/ByPlaceOfBirth.xlsx')
read_file.to_csv ('americabirthplace.csv', index = None, header=True)