# 这个文件存储的都是自己写的RNN模型
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size,output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input,hidden):
        combined = torch.cat((input,hidden),1)      # 使用torch.cat将input和hidden按列拼接
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)

# 情感分析的模型
class SenNet(nn.Module):
    def __init__(self,embed_size,num_hiddens,num_layers,
                 weight,out_size):
        super(SenNet,self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False         # 禁止训练时对词向量进行微调

        #数据向量大小等于词向量大小,hidden_size:隐藏元的个数,num_layers:隐藏元的层数
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=self.num_hiddens,num_layers=num_layers)
        self.h2o = nn.Linear(num_hiddens*2,out_size)

    def forward(self, input):
        # 这里input大小是 [b,m],b个序列，每个序列有m个词
        # 然后使用embedding 将其转为[b,m,n],其中n为词向量的维度
        embeddings = self.embedding(input)

        # lstm返回三个参数,output,(hn,cn),其中output是最后一层lstm的每个词向量对应隐藏层的输出，与input大小次昂等
        # hn,cn是所有层最后一个隐藏元和记忆元的输出
        out,h = self.lstm(embeddings.permute([1,0,2]))    # 这里为了放进lstm，将其转为[m,b,n]

        # 全连接层时传入out的多少行
        # out = self.h2o(torch.cat([o for o in out],dim=1))
        # out = self.h2o(torch.cat([out[0],out[len(out)//2],out[-1]],dim=1))
        out = self.h2o(torch.cat([out[0],out[-1]],dim=1))


        return out

# 预测sin函数的模型
class SinNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(SinNet,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm1 = nn.LSTMCell(input_size=input_size,hidden_size=hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=hidden_size,hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size,1)

    def forward(self, input):
        outputs = []
        # 根据用户输入的input_size选择使用多少个前面的点预测后面的一个点
        # 比如input_size = 3,则用 n-2,n-1,n -> n+1
        # 如果input_size = 1,则为 n -> n+1

        for i in range(input.size(1)-self.input_size+1):
            ht,ct = self.lstm1(input[:,i:i+self.input_size])
            ht,ct = self.lstm2(ht)
            output = self.linear(ht)
            outputs+=[output]
        outputs = torch.stack(outputs,1).squeeze(2)
        return outputs

    def predict(self,input,predict_num):
        """
        用户传入input和要预测的个数，返回给用户预测值
        :param input(tensor[n,m]):预测数据的依据
        :param predict_num(int):被预测出的数据个数
        :return(tensor[n,m+predict_num]):根据input预测出来的数据
        """
        outputs = self.forward(input)
        for i in range(predict_num):
            ht,ct = self.lstm1(outputs[:,-self.input_size:])
            ht,ct = self.lstm2(ht)
            output = self.linear(ht)
            outputs = torch.cat([outputs,output],dim=1)

        return outputs

