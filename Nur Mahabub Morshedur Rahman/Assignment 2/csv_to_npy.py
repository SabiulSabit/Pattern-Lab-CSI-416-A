import pandas as pd
import numpy as np

data = pd.read_csv("inputFIlePath", header=None)
data = pd.DataFrame(data)
data.drop([0],inplace=True,axis=1) #delete 1st column
np.save('outPutFilePath',data)