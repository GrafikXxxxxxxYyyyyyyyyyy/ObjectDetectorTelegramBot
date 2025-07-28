import os
import torch
import asyncio
import random
from datetime import datetime
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup, 
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes,
    CallbackQueryHandler,
)
from transformers import (
    AutoImageProcessor, 
    DFineForObjectDetection,
    AutoModelForObjectDetection,
)
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# 🔧 Модели
DETECTOR_MODELS = {
    "detr": {
        "name": "DETR ResNet-101",
        "desc": "🎯 DETR (ResNet-101) — мощная модель от Facebook для детекции объектов. Подходит для общего применения.",
        "processor": AutoImageProcessor.from_pretrained("facebook/detr-resnet-101"),
        "model": AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-101").to('mps')
    },
    "yolo": {
        "name": "YOLOv8 Nano",
        "desc": "⚡ YOLOv8n — очень быстрая модель для реального времени. Подходит для мобильных и слабых устройств.",
        "model": YOLO('yolov8n.pt').to('mps')  # YOLO не использует processor
    },
    "dfine": {
        "name": "DFine XLarge",
        "desc": "🔬 DFine XLarge — современная модель для тонкой детекции объектов. Подходит для сложных сцен.",
        "processor": AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj365"),
        "model": DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj365").to('mps')
    }
}



def detect_and_draw_boxes(image: Image.Image, model_key: str) -> tuple[Image.Image, str]:
    found_objects = []
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if model_key == "yolo":
        results = DETECTOR_MODELS["yolo"]["model"](image)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                label = DETECTOR_MODELS["yolo"]["model"].names[cls]
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                text = f"{label} ({score:.2f})"
                found_objects.append(text)
                draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
                draw.text((x0 + 3, y0 + 3), text, fill="green", font=font)

    else:
        processor = DETECTOR_MODELS[model_key]["processor"]
        model = DETECTOR_MODELS[model_key]["model"]
        inputs = processor(images=image, return_tensors="pt").to("mps")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to("mps")
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 3) for i in box.tolist()]
            x0, y0, x1, y1 = box
            class_name = model.config.id2label[label.item()]
            text = f"{class_name} ({score:.2f})"
            found_objects.append(text)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0 + 3, y0 + 3), text, fill="red", font=font)

    # Формируем текст
    if found_objects:
        object_list_text = "🔍 Найденные объекты:\n" + "\n".join(f"• {obj}" for obj in found_objects)
    else:
        object_list_text = "Ничего не найдено 😔"

    return image, object_list_text



# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [["Detect objects"], ["Change detector model"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Привет! Нажми кнопку ниже, чтобы запустить детекцию объектов.", reply_markup=reply_markup)



# Обработчик нажатия кнопки
async def detect_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    if text == "Detect objects":
        context.user_data["awaiting_image"] = True
        await update.message.reply_text("Окей, жду ваше изображение 📷")


# Обработчик смены модели детектора
async def change_model_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("DETR ResNet-101", callback_data="set_model_detr")],
        [InlineKeyboardButton("YOLOv8 Nano", callback_data="set_model_yolo")],
        [InlineKeyboardButton("DFine XLarge", callback_data="set_model_dfine")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    description_text = "\n\n".join([f"*{v['name']}*\n{v['desc']}" for v in DETECTOR_MODELS.values()])
    await update.message.reply_text(
        f"Выберите модель для детекции объектов:\n\n{description_text}",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )


async def set_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    model_key = query.data.replace("set_model_", "")
    context.user_data["current_model"] = model_key

    model_name = DETECTOR_MODELS[model_key]["name"]
    await query.edit_message_text(f"✅ Модель *{model_name}* выбрана.", parse_mode="Markdown")




##########################################################################################
# Обработка изображений
##########################################################################################
async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_image", False):
        await update.message.reply_text("Сначала нажмите кнопку '🔍 Обнаружить объекты'.")
        return

    context.user_data["awaiting_image"] = False  # Сброс флага
    await update.message.reply_text("Фото получено. Обрабатываю 🧠...")
    
    # 1. Получаем файл
    photo = update.message.photo[-1] # Получаем самую большую (по размеру) версию изображения
    file = await context.bot.get_file(photo.file_id)  
    photo_bytes = await file.download_as_bytearray() # получаем фото в виде битового массива
    image = Image.open(BytesIO(photo_bytes)).convert("RGB") # открываем в формате PIL

    # Обработка изображения
    model_key = context.user_data.get("current_model", "detr")
    result_image, detected_text = detect_and_draw_boxes(image, model_key)

    # 5. Сохраняем изображение в буфер без потери качества
    buffer = BytesIO()
    result_image.save(buffer, format="JPEG", quality=100, subsampling=0)
    buffer.seek(0)

    # Сохраняем локально
    os.makedirs("photos", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photos/result_{timestamp}.jpg"
    result_image.save(filename, format="JPEG", quality=100, subsampling=0)
    
    await update.message.reply_photo(photo=buffer, caption=detected_text)
##########################################################################################



def main():
    # Вставь сюда свой токен от BotFather
    TOKEN = "8296197576:AAGRlQqDeNbGF1rWY8f8SXFIONubN4EIycQ"

    # Приложение самого бота
    app = ApplicationBuilder().token(TOKEN).build()

    # Обработка команды /start
    app.add_handler(CommandHandler("start", start))
    # Обработка нажатия кнопок
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Detect objects$"), detect_button_handler))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Change detector model$"), change_model_handler))
    app.add_handler(CallbackQueryHandler(set_model_callback, pattern=r"^set_model_"))
    # Обработка фото
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))

    # Запуск бота
    print("Бот запущен. Нажми Ctrl+C для остановки.")
    app.run_polling()



if __name__ == "__main__":
    main()

