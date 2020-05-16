# coding=utf-8

import torch
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import torch.onnx
import onnxruntime

import sys
sys.path.append("../..")

from core.classifier import SoftmaxClassifier

if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = np.array(iris.data), np.array(iris.target)
    x_train, x_test, y_train, y_test = [torch.from_numpy(item) for item in train_test_split(X,y,test_size=0.1)]
    print(x_train)
    print(y_train)
    
    model = SoftmaxClassifier(num_feature=4, num_class=3)
    optim = torch.optim.SGD(model.parameters(),lr=0.6)
    for epoch in range(1, 10001):
        model.train()
        loss = model.loss(x_train.float(), y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch %  100 == 0:
            model.eval()
            dev_loss = model.loss(x_test.float(), y_test)
            print("epoch", epoch, " loss:", loss, ", dev loss:", dev_loss)

    y_hat = model(x_test.float())
    for prob, label in zip(y_hat, y_test):
        print(prob, label)

    # 用torch预测[6.2, 3.4, 5.4, 2.3], 比较精度
    y_hat = model(torch.tensor([[6.2, 3.4, 5.4, 2.3]]))
    print(y_hat)

    # 保存模型    
    torch.onnx.export(model,                   # model being run
                      torch.tensor([[6.2, 3.4, 5.4, 2.3]]),               # model input (or a tuple for multiple inputs)
                      "./softmax.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,      # store the trained parameter weights inside the model file
                      input_names = ['input'], # the model's input names
                      output_names = ['output']# the model's output names
    ) 
    


    # 用onnx预测[6.2, 3.4, 5.4, 2.3], 比较精度
    ort_session = onnxruntime.InferenceSession("./softmax.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: [[6.2, 3.4, 5.4, 2.3]]}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
    input_name_X = ort_session.get_inputs()[0].name 
    print(input_name_X, " type: ", ort_session.get_inputs()[0].type) 
