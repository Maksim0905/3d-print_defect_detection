import cv2
import time
import torch
import telebot
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import threading
from torchvision import transforms
from requests.exceptions import ConnectionError, ReadTimeout
import sys
import os


print_status = True
predictions_window = []
bot = telebot.TeleBot('7053382998:AAHybbMj0J_ON9aM7kISp7ni5jcMexlGRQ0')


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


MODEL_PATH = "defect_detection_model.pth"
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def predict_label(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = 'good' if predicted.item() == 1 else 'bad'
    return label


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(
        message.chat.id, "Привет! Для начала процесса отслеживания печати введите команду /track.")


@bot.message_handler(commands=['track'])
def handle_track(message):
    bot.send_message(
        message.chat.id, "Начинаю отслеживание печати. Ожидайте результатов...")

    global predictions_window
    predictions_window = []

    global print_status
    while print_status:
        frame = cv2.VideoCapture(0).read()[1]
        image = Image.fromarray(frame)
        preprocessed_image = preprocess_image(
            image)
        label = predict_label(preprocessed_image)
        # print(label)
        predictions_window.append(label)

        if len(predictions_window) > 20:
            bad_predictions_count = sum(
                1 for prediction in predictions_window[-20:] if prediction == 'bad')
            if bad_predictions_count > 10:
                bot.send_message(
                    message.chat.id, f'Печать приостановлена из-за обнаружения дефекта.\nВероятность деффекта {len(predictions_window[-20:]) / bad_predictions_count * 100}')
                break


@bot.message_handler(commands=['stop'])
def handle_stop(message):
    global print_status
    print_status = False
    bad_predictions_count = sum(
        1 for prediction in predictions_window[-20:] if prediction == 'bad')
    bot.send_message(message.chat.id, "Печать остановлена.")
    bot.send_message(message.chat.id, "Результаты отслеживания печати:")
    bot.send_message(message.chat.id, "Количество дефектов: " +
                     str(bad_predictions_count))
    bot.send_message(message.chat.id, "Количество хороших результатов: " +
                     str(len(predictions_window[-20:]) - predictions_window.count('good')))
    bot.send_message(
        message.chat.id, "Процесс отслеживания печати остановлен.")


try:
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
except (ConnectionError, ReadTimeout) as e:
    sys.stdout.flush()
    os.execv(sys.argv[0], sys.argv)
else:
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
