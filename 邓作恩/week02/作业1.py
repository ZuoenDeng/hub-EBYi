import pandas as pd
import torch
import torch.nn as nn # 导入PyTorch神经网络模块，提供构建神经网络所需的各种层和激活函数等组件
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 加载数据
dataset = pd.read_csv('dataset.csv', sep='\t', header=None)
texts = dataset[0].tolist()
labels = dataset[1].tolist()

# 创建字典：标签-索引，为所有标签创建唯一的索引
label_to_index = {label: i for i, label in enumerate(set(labels))} # 将标签映射为索引,label_to_index是字典
numerical_labels = [label_to_index[label] for label in labels]

# 构建字典：字符-索引，为所有文本中的字符创建唯一的索引
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
# 创建字典：索引-字符，反向索引
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index) # 词汇表大小
print('词汇表大小：', vocab_size)

# 分词统一的长度
max_len = 40

class CharBowDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = labels
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()
        print(len(texts), '条数据  ',  len(labels), '条标签：')

    def _create_bow_vectors(self): # 创建词袋向量——将每一句文本转换为一句统计各字符频次的向量

        # 将每一句文本中的字符转换为索引，并统一长度
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        # 将每一句文本的字符索引转换为词频向量——即词袋向量
        bow_vectors = []
        for tokenized in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size) # 初始化一个全零向量
            for token in tokenized:
                if token != 0:
                    bow_vector[token] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

#  创建网络模型
class SimplClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 创建数据集和数据加载器
char_dataset = CharBowDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 超参数设置和模型实例化
hidden_dim = [128]
output_dim = len(label_to_index)
modle = SimplClassifier(vocab_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modle.parameters(), lr=0.0001) # Adam收敛的更快

num_epochs = 10
losses = []
for epoch in range(num_epochs):
    modle.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = modle(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(losses)
plt.show()

def classify_text(text, char_to_index, max_len, vocab_size, model, index_to_label):
    # 第一步：分词
    tokenized = [char_to_index.get(char, 0) for char in text]
    tokenized += [0] * (max_len - len(tokenized))

    # 第二步：创建词袋向量——将每一句文本转换成一句统计各字符频次的向量
    bow_vector = torch.zeros(vocab_size)
    for token in tokenized:
        if token != 0:
            bow_vector[token] += 1
    bow_vector = bow_vector.unsqueeze(0) # 添加一个批量维度

    model.eval()
    with torch.no_grad():
        outputs = model(bow_vector)

    _, predicted = torch.max(outputs, 1) # 获取预测结果
    predicted_label = index_to_label[predicted.item()]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_label = classify_text(new_text, char_to_index, max_len, vocab_size, modle, index_to_label)
print(f'预测结果：{predicted_label}')

new_text = "帮我播放张杰的歌"
predicted_label = classify_text(new_text, char_to_index, max_len, vocab_size, modle, index_to_label)
print(f'预测结果：{predicted_label}')