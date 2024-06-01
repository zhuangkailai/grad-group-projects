import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import onnx
import onnx.utils

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))
        return output



# 创建一个模型实例
model = TeacherNet()
model2 = StudentNet()
model3 = StudentNet()
# 加载模型权重
model.load_state_dict(torch.load('teacher.pt'))
model2.load_state_dict(torch.load('student.pt'))
model3.load_state_dict(torch.load('student_kd.pt'))
# 设置模型为评估模式
model.eval()
model2.eval()
model3.eval()


# 定义一个虚拟输入
dummy_input = torch.randn(1, 1, 28, 28)  # 假设输入大小为[1, 1, 28, 28]

# 导出模型到ONNX格式
onnx_path = "teacher_net.onnx"
onnx_path2 = "student_net.onnx"
onnx_path3 = "student_kd_net.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
torch.onnx.export(model2, dummy_input, onnx_path2, verbose=True)
torch.onnx.export(model3, dummy_input, onnx_path3, verbose=True)

#print(f"Model exported as {onnx_path}")
