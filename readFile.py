import pandas as pd
import numpy as np

def read_csv():
  zc_dataframe = pd.read_csv("bankdata.csv", sep=",", header=0)
  totalNum = zc_dataframe.shape[0]
  inputNum = int(totalNum*0.8)
  outputNum = totalNum - inputNum
  print("数据总量：", totalNum)
  print("训练量：", inputNum)
  print("预测量", outputNum)
  X_assess = []
  Y_assess = []
  X_test_assess = []
  Y_test_assess = []
  for zc_index in range(inputNum):
      zc_row = zc_dataframe.values[zc_index]
      zc_rowArray = zc_row.tolist()
      tempX_1 = np.delete(zc_row, -1, axis=0)
      X_assess.append(tempX_1.tolist())
      Y_assess.append([zc_rowArray[20]])

  for testIndex in range(outputNum):
      zc_row = zc_dataframe.values[testIndex + inputNum]
      zc_rowArray = zc_row.tolist()
      tempX_1 = np.delete(zc_row, -1, axis=0)
      X_test_assess.append(tempX_1.tolist())
      Y_test_assess.append([zc_rowArray[20]])
  return (X_assess, Y_assess, X_test_assess, Y_test_assess)
