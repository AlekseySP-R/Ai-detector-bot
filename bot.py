import os
import logging
import tempfile
import threading
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from ai_detector import AIContentDetector, format_result

# --- Flask (для Render) ---
app = Flask(__name__)

@app.route('/')
def index():
    return "Бот работает!"

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
# ---------------------------

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
        
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.app.add_handler(MessageHandler(filters.Document.IMAGE, self.handle_image_document))
        self.app.add_handler(MessageHandler(filters.VIDEO, self.handle_video))
        self.app.add_handler(MessageHandler(filters.Document.VIDEO, self.handle_video_document))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio))
        self.app.add_handler(MessageHandler(filters.Document.AUDIO, self.handle_audio_document))
        
        self.stats = {"images": 0, "videos": 0, "audio": 0, "total": 0}
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = "🤖 *Детектор AI-контента*\n\nОтправьте файл (фото, видео, аудио) для проверки."
        await update.message.reply_text(msg)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("📚 Пришлите файл, и я проверю его на признаки AI.")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s = self.stats
        msg = (f"📊 Статистика:\n🖼 Картинок: {s['images']}\n"
               f"🎥 Видео: {s['videos']}\n🎵 Аудио: {s['audio']}\n"
               f"📈 Всего: {s['total']}")
        await update.message.reply_text(msg)

    async def _safe_process(self, update, file_obj, ext, media_type, detect_func):
        tmp_path = None
        try:
            await update.message.reply_text(f"🔍 Анализирую {media_type}...")
            
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
            
            await file_obj.download_to_drive(tmp_path)
            
            # Получаем результат
            result = detect_func(tmp_path)
            
            # Форматируем ответ (без Markdown, чтобы не было ошибок)
            text = format_result(result)
            
            # Отправляем ответ
            await update.message.reply_text(text)
            
            self.stats[media_type] += 1
            self.stats['total'] += 1

        except Exception as e:
            logger.error(f"Error: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        photo = update.message.photo[-1]
        file = await photo.get_file()
        await self._safe_process(update, file, '.jpg', 'images', self.detector.detect_image)

    async def handle_image_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        ext = os.path.splitext(doc.file_name)[1].lower()
        file = await doc.get_file()
        await self._safe_process(update, file, ext, 'images', self.detector.detect_image)

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        vid = update.message.video
        if vid.file_size > 20 * 1024 * 1024:
            await update.message.reply_text("⚠️ Видео до 20 МБ.")
            return
        file = await vid.get_file()
        await self._safe_process(update, file, '.mp4', 'videos', self.detector.detect_video)

    async def handle_video_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        if doc.file_size > 20 * 1024 * 1024:
            await update.message.reply_text("⚠️ Видео до 20 МБ.")
            return
        ext = os.path.splitext(doc.file_name)[1].lower()
        file = await doc.get_file()
        await self._safe_process(update, file, ext, 'videos', self.detector.detect_video)

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message.voice:
            audio = update.message.voice
            ext = '.ogg'
        else:
            audio = update.message.audio
            ext = '.mp3'
        file = await audio.get_file()
        await self._safe_process(update, file, ext, 'audio', self.detector.detect_audio)

    async def handle_audio_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        ext = os.path.splitext(doc.file_name)[1].lower()
        file = await doc.get_file()
        await self._safe_process(update, file, ext, 'audio', self.detector.detect_audio)

    def run(self):
        logger.info("Бот запущен!")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ Нет токена TELEGRAM_BOT_TOKEN!")
        exit(1)
        
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    
    bot = AIDetectorBot(TOKEN)
    bot.run()
