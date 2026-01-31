import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 读取文本与标签
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 创建标签索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符索引
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
index_to_char = {i: char for char, i in char_to_index.items()} # 逆序字符索引
vocab_size = len(char_to_index) # 词表大小

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]] # 将字符转换为索引，从而得到数字化的文本表示
        indices += [0] * (self.max_len - len(indices)) # 补长
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        rnn_out, hidden_state = self.rnn(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0)) # 获取最后一层输出
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0)) # 获取最后一层输出
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, hidden_state = self.gru(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0)) # 获取最后一层输出
        return out

# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model_rnn = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model_lstm = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model_gru = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)
optimizer_gru = optim.Adam(model_gru.parameters(), lr=0.001)

losses_rnn = []
losses_lstm = []
losses_gru = []
num_epochs = 10
for epoch in range(num_epochs):
    model_rnn.train()
    model_lstm.train()
    model_gru.train()
    # running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_rnn.zero_grad()
        model_lstm.zero_grad()
        model_gru.zero_grad()
        outputs_rnn = model_rnn(inputs)
        outputs_lstm = model_lstm(inputs)
        outputs_gru = model_gru(inputs)
        loss_rnn = criterion(outputs_rnn, labels)
        loss_lstm = criterion(outputs_lstm, labels)
        loss_gru = criterion(outputs_gru, labels)
        loss_rnn.backward()
        loss_lstm.backward()
        loss_gru.backward()
        optimizer_rnn.step()
        optimizer_lstm.step()
        optimizer_gru.step()
        # running_loss += loss_rnn.item()
        losses_rnn.append(loss_rnn.item())
        losses_lstm.append(loss_lstm.item())
        losses_gru.append(loss_gru.item())
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss_rnn.item(), loss_lstm.item(), loss_gru.item()}")

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

plt.plot(losses_rnn,  label="RNN Loss")
plt.plot(losses_lstm, label="LSTM Loss")
plt.plot(losses_gru, label="GRU Loss")
plt.legend()
plt.show()

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class_rnn = classify_text_lstm(new_text, model_rnn, char_to_index, max_len, index_to_label)
predicted_class_lstm = classify_text_lstm(new_text, model_lstm, char_to_index, max_len, index_to_label)
predicted_class_gru = classify_text_lstm(new_text, model_gru, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class_rnn, predicted_class_lstm, predicted_class_gru}'")

new_text_2 = "查询明天北京的天气"
predicted_class_rnn_2 = classify_text_lstm(new_text_2, model_rnn, char_to_index, max_len, index_to_label)
predicted_class_lstm_2 = classify_text_lstm(new_text_2, model_lstm, char_to_index, max_len, index_to_label)
predicted_class_gru_2 = classify_text_lstm(new_text_2, model_gru, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_rnn_2, predicted_class_lstm_2, predicted_class_gru_2}'")