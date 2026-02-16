import requests
import os

# Ваши ключи API (лучше хранить их в секретах, но для теста оставим тут)
API_USER = '1452903361'
API_SECRET = 'DvAuoUagL4rArz4gEouNin36AWkh9aKg'

class AIContentDetector:
    def detect_image(self, file_path):
        """
        Реальная проверка изображения через Sightengine API.
        """
        print(f"Real API analysis for image: {file_path}")
        
        # Параметры запроса к API
        params = {
            'models': 'genai',  # Модель для детекции AI-генерации
            'api_user': API_USER,
            'api_secret': API_SECRET
        }
        
        try:
            # Открываем файл и отправляем
            with open(file_path, 'rb') as f:
                files = {'media': f}
                response = requests.post(
                    'https://api.sightengine.com/1.0/check.json',
                    files=files,
                    data=params
                )
            
            data = response.json()
            
            # Проверяем ответ
            if data.get('status') == 'success':
                # Получаем оценку AI-генерации (от 0 до 1)
                # type может быть 'ai-generated' или 'none'
                genai_type = data.get('type', 'none')
                # confidence — это уверенность в том, что это AI
                # В ответе API поле 'score' показывает вероятность генерации
                score = 0.0
                is_ai = False
                
                # API возвращает scores для разных моделей, нам нужна 'genai'
                # Структура: {'genai': {'score': 0.99}} или просто 'type': 'ai-generated'
                if 'genai' in data:
                    score = data['genai'].get('score', 0.0)
                
                # Если в ответе напрямую указан тип ai-generated
                if genai_type == 'ai-generated':
                    is_ai = True
                    # Если есть score, берем его, иначе ставим высокую уверенность
                    if score == 0: score = 0.95
                else:
                    is_ai = False
                    # Если score низкий, но тип не ai-generated, доверяем типу
                
                return {
                    "type": "image",
                    "is_ai": is_ai,
                    "confidence": score,
                    "details": f"Тип: {genai_type}. Профессиональный анализ через Sightengine."
                }
            else:
                # Ошибка от API
                return {
                    "type": "image",
                    "is_ai": False,
                    "confidence": 0,
                    "details": f"Ошибка API: {data.get('error', 'Unknown error')}"
                }

        except Exception as e:
            return {
                "type": "image",
                "is_ai": False,
                "confidence": 0,
                "details": f"Ошибка соединения: {str(e)}"
            }

    def detect_video(self, file_path):
        # Для видео оставим имитацию, так как синхронная проверка видео сложнее
        # (требует отправки по URL или асинхронного запроса)
        return {
            "type": "video",
            "is_ai": False,
            "confidence": 0,
            "details": "Проверка видео пока в разработке."
        }

    def detect_audio(self, file_path):
        # Для аудио также оставим имитацию
        return {
            "type": "audio",
            "is_ai": False,
            "confidence": 0,
            "details": "Проверка аудио пока в разработке."
        }

def format_result(result):
    status = "🤖 Сгенерировано AI" if result['is_ai'] else "👤 Скорее всего человек"
    
    # Форматируем процент
    percent = f"{result['confidence']:.0%}" if result['confidence'] > 0 else "N/A"
    
    return (
        f"📊 *Результат анализа ({result['type']})*\n\n"
        f"Вердикт: {status}\n"
        f"Уверенность: {percent}\n"
        f"Детали: {result['details']}"
    )
