"""
Telegram Bot для детектора AI-контента
"""

import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from ai_detector import AIContentDetector, format_result
import tempfile

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class AIDetectorBot:
    def __init__(self, token: str):
        self.token = token
        self.detector = AIContentDetector()
        self.app = Application.builder().token(token).build()
        
        # Регистрация обработчиков
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        
        # Обработчики файлов
        self.app.add_handler(MessageHandler(
            filters.PHOTO, self.handle_photo
        ))
        self.app.add_handler(MessageHandler(
            filters.Document.IMAGE, self.handle_image_document
        ))
        self.app.add_handler(MessageHandler(
            filters.VIDEO, self.handle_video
        ))
        self.app.add_handler(MessageHandler(
            filters.Document.VIDEO, self.handle_video_document
        ))
        self.app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO, self.handle_audio
        ))
        self.app.add_handler(MessageHandler(
            filters.Document.AUDIO, self.handle_audio_document
        ))
        
        # Статистика
        self.stats = {
            "images": 0,
            "videos": 0,
            "audio": 0,
            "total": 0
        }
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /start"""
        welcome_message = """
🤖 *Детектор AI-контента*

Привет! Я помогу вам определить, создан ли контент с помощью искусственного интеллекта.

📤 *Что я могу анализировать:*
• 🖼 Изображения (фото, картинки)
• 🎥 Видео
• 🎵 Аудио (голос, музыка)

📝 *Как использовать:*
Просто отправьте мне файл, и я проанализирую его!

⚡️ *Команды:*
/help - Помощь и примеры
/stats - Статистика использования

🔬 *Что я проверяю:*
• Метаданные файла
• Паттерны и артефакты
• Спектральные характеристики
• Признаки AI-генерации

⚠️ Помните: результат - это вероятностная оценка, не 100% гарантия.
"""
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /help"""
        help_message = """
📚 *Руководство по использованию*

*🖼 Изображения:*
Отправьте фото или изображение как документ.
Я проверю:
  • Метаданные (EXIF)
  • Паттерны шума
  • Частотные характеристики
  • Типичные AI-артефакты

*🎥 Видео:*
Отправьте видеофайл.
Я проверю:
  • Отдельные кадры
  • Темпоральную согласованность
  • AI-артефакты в движении

*🎵 Аудио:*
Отправьте голосовое сообщение или аудиофайл.
Я проверю:
  • Спектральные характеристики
  • Естественность звучания
  • Паттерны синтеза

*📊 Интерпретация результатов:*
🔴 70-100% - Вероятно создано AI
🟡 40-69% - Возможно создано AI
🟢 0-39% - Вероятно создано человеком

*💡 Советы:*
• Для лучших результатов отправляйте оригинальные файлы
• Избегайте сжатых/пересохраненных версий
• Большие файлы могут обрабатываться дольше

Вопросы? Напишите @your_support
"""
        await update.message.reply_text(
            help_message,
            parse_mode='Markdown'
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать статистику"""
        stats_message = f"""
📊 *Статистика бота*

🖼 Изображений проанализировано: {self.stats['images']}
🎥 Видео проанализировано: {self.stats['videos']}
🎵 Аудио проанализировано: {self.stats['audio']}

📈 Всего файлов: {self.stats['total']}

Спасибо, что используете наш сервис! 🙏
"""
        await update.message.reply_text(
            stats_message,
            parse_mode='Markdown'
        )
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка фотографий"""
        await update.message.reply_text("🔍 Анализирую изображение...")
        
        try:
            # Получение файла
            photo = update.message.photo[-1]  # Берем максимальное разрешение
            file = await photo.get_file()
            
            # Сохранение во временный файл
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            # Анализ
            result = self.detector.detect_image(tmp_path)
            formatted_result = format_result(result)
            
            # Отправка результата
            await update.message.reply_text(formatted_result)
            
            # Удаление временного файла
            os.unlink(tmp_path)
            
            # Обновление статистики
            self.stats['images'] += 1
            self.stats['total'] += 1
            
        except Exception as e:
            logger.error(f"Error processing photo: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке: {str(e)}"
            )
    
    async def handle_image_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка изображений как документов"""
        await update.message.reply_text("🔍 Анализирую изображение...")
        
        try:
            document = update.message.document
            file = await document.get_file()
            
            # Проверка расширения
            file_ext = os.path.splitext(document.file_name)[1].lower()
            if file_ext not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                await update.message.reply_text(
                    "❌ Неподдерживаемый формат изображения. "
                    "Используйте: JPG, PNG, WEBP, BMP"
                )
                return
            
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            result = self.detector.detect_image(tmp_path)
            formatted_result = format_result(result)
            
            await update.message.reply_text(formatted_result)
            os.unlink(tmp_path)
            
            self.stats['images'] += 1
            self.stats['total'] += 1
            
        except Exception as e:
            logger.error(f"Error processing image document: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке: {str(e)}"
            )
    
    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка видео"""
        await update.message.reply_text(
            "🔍 Анализирую видео... Это может занять некоторое время ⏳"
        )
        
        try:
            video = update.message.video
            file = await video.get_file()
            
            # Проверка размера (ограничение в 20 МБ для Telegram API)
            if video.file_size > 20 * 1024 * 1024:
                await update.message.reply_text(
                    "⚠️ Видео слишком большое. Максимальный размер: 20 МБ"
                )
                return
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            result = self.detector.detect_video(tmp_path)
            formatted_result = format_result(result)
            
            await update.message.reply_text(formatted_result)
            os.unlink(tmp_path)
            
            self.stats['videos'] += 1
            self.stats['total'] += 1
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке: {str(e)}"
            )
    
    async def handle_video_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка видео как документа"""
        await update.message.reply_text(
            "🔍 Анализирую видео... Это может занять некоторое время ⏳"
        )
        
        try:
            document = update.message.document
            file = await document.get_file()
            
            file_ext = os.path.splitext(document.file_name)[1].lower()
            if file_ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                await update.message.reply_text(
                    "❌ Неподдерживаемый формат видео. "
                    "Используйте: MP4, AVI, MOV, MKV, WEBM"
                )
                return
            
            if document.file_size > 50 * 1024 * 1024:
                await update.message.reply_text(
                    "⚠️ Видео слишком большое. Максимальный размер: 50 МБ"
                )
                return
            
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            result = self.detector.detect_video(tmp_path)
            formatted_result = format_result(result)
            
            await update.message.reply_text(formatted_result)
            os.unlink(tmp_path)
            
            self.stats['videos'] += 1
            self.stats['total'] += 1
            
        except Exception as e:
            logger.error(f"Error processing video document: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке: {str(e)}"
            )
    
    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка аудио/голоса"""
        await update.message.reply_text("🔍 Анализирую аудио...")
        
        try:
            # Обработка голосового сообщения или аудио
            if update.message.voice:
                audio = update.message.voice
                file_ext = '.ogg'
            else:
                audio = update.message.audio
                file_ext = '.mp3'
            
            file = await audio.get_file()
            
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            result = self.detector.detect_audio(tmp_path)
            formatted_result = format_result(result)
            
            await update.message.reply_text(formatted_result)
            os.unlink(tmp_path)
            
            self.stats['audio'] += 1
            self.stats['total'] += 1
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке: {str(e)}"
            )
    
    async def handle_audio_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка аудио как документа"""
        await update.message.reply_text("🔍 Анализирую аудио...")
        
        try:
            document = update.message.document
            file = await document.get_file()
            
            file_ext = os.path.splitext(document.file_name)[1].lower()
            if file_ext not in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
                await update.message.reply_text(
                    "❌ Неподдерживаемый формат аудио. "
                    "Используйте: MP3, WAV, OGG, M4A, FLAC"
                )
                return
            
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                tmp_path = tmp.name
            
            result = self.detector.detect_audio(tmp_path)
            formatted_result = format_result(result)
            
            await update.message.reply_text(formatted_result)
            os.unlink(tmp_path)
            
            self.stats['audio'] += 1
            self.stats['total'] += 1
            
        except Exception as e:
            logger.error(f"Error processing audio document: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке: {str(e)}"
            )
    
    def run(self):
        """Запуск бота"""
        logger.info("Бот запущен!")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # Получение токена из переменной окружения
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not TOKEN:
        print("❌ Ошибка: Не найден токен бота!")
        print("Установите переменную окружения TELEGRAM_BOT_TOKEN")
        print("\nПример:")
        print("export TELEGRAM_BOT_TOKEN='ваш_токен_здесь'")
        exit(1)
    
    bot = AIDetectorBot(TOKEN)
    bot.run()
