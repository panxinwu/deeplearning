import numpy as np

class NeuralNetwork():
    def __init__(self):
        # 随机数种子
        np.random.seed(1)
        #产生 -1~1 的 3*1随机数矩阵
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
    def sigmoid(self, x):
        #定义sigmoid方法
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        #计算sigmoid函数的导数
        return x * (1 - x)
    def train(self, training_inputs, training_outputs, training_iterations):
        #训练模型  不断调整权重同时进行进准预测
        for iteration in range(training_iterations):
            #通过神经元接收训练数据
            output = self.think(training_inputs)
            #反向传播误差率的计算
            error = training_outputs - output
            #权重调整
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments
    def think(self, inputs):
        #通过神经元传递输入以获得输出
        #将值转换为浮点数
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
if __name__ == "__main__":
    #实例化 NeuralNetwork 类
    neural_network = NeuralNetwork()
    print("开始随机生成权重: ")
    print(neural_network.synaptic_weights)

    #训练数据包含 4个例子 每个例子包含：3个输入值和1个输出值

    training_inputs = np.array([[0,0,1],
     [1,1,1],
     [1,0,1],
     [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T
    #正式开始训练
    neural_network.train(training_inputs, training_outputs, 15000)
    print("训练结束后的权重: ")
    print(neural_network.synaptic_weights)
    # 训练结束后开始预测 输入三个预测值
    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))
    print("新的训练值: ", user_input_one, user_input_two, user_input_three)
    print("新的预测数据为: ")
    thinkData = neural_network.think(np.array([user_input_one, user_input_two, user_input_three]))
    print(thinkData)
    print("神经网模型预测准确率为:")
    print(str((thinkData[0]/1)*100) + '%')
