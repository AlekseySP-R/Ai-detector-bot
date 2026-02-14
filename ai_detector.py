"""
AI Content Detector
Детектор AI-сгенерированного контента (изображения, видео, аудио)
"""

import cv2
import numpy as np
from PIL import Image
import librosa
import io
from typing import Dict, Tuple
import hashlib


class AIContentDetector:
    """Детектор AI-сгенерированного контента"""
    
    def __init__(self):
        self.image_features = []
        self.audio_features = []
        
    def detect_image(self, image_path: str) -> Dict:
        """
        Анализирует изображение на наличие признаков AI-генерации
        
        Returns:
            Dict с результатами анализа
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Не удалось загрузить изображение"}
            
            results = {
                "type": "image",
                "ai_probability": 0.0,
                "indicators": [],
                "details": {}
            }
            
            # 1. Анализ метаданных (EXIF)
            metadata_score = self._analyze_metadata(image_path)
            results["details"]["metadata_analysis"] = metadata_score
            
            # 2. Анализ шума и артефактов
            noise_score = self._analyze_noise_patterns(img)
            results["details"]["noise_patterns"] = noise_score
            
            # 3. Анализ симметрии и паттернов
            symmetry_score = self._analyze_symmetry(img)
            results["details"]["symmetry_analysis"] = symmetry_score
            
            # 4. Анализ частотных характеристик
            frequency_score = self._analyze_frequency(img)
            results["details"]["frequency_analysis"] = frequency_score
            
            # 5. Проверка на типичные артефакты GAN/Diffusion
            artifact_score = self._detect_ai_artifacts(img)
            results["details"]["ai_artifacts"] = artifact_score
            
            # Расчет итоговой вероятности
            weights = {
                "metadata": 0.15,
                "noise": 0.25,
                "symmetry": 0.15,
                "frequency": 0.20,
                "artifacts": 0.25
            }
            
            total_score = (
                metadata_score * weights["metadata"] +
                noise_score * weights["noise"] +
                symmetry_score * weights["symmetry"] +
                frequency_score * weights["frequency"] +
                artifact_score * weights["artifacts"]
            )
            
            results["ai_probability"] = round(total_score * 100, 2)
            
            # Добавление индикаторов
            if metadata_score > 0.6:
                results["indicators"].append("Отсутствие/подозрительные метаданные")
            if noise_score > 0.7:
                results["indicators"].append("Неестественное распределение шума")
            if symmetry_score > 0.6:
                results["indicators"].append("Подозрительные паттерны симметрии")
            if frequency_score > 0.7:
                results["indicators"].append("Аномалии в частотном спектре")
            if artifact_score > 0.7:
                results["indicators"].append("Обнаружены AI-артефакты")
            
            return results
            
        except Exception as e:
            return {"error": f"Ошибка анализа: {str(e)}"}
    
    def detect_video(self, video_path: str) -> Dict:
        """Анализирует видео на наличие признаков AI-генерации"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Не удалось открыть видео"}
            
            results = {
                "type": "video",
                "ai_probability": 0.0,
                "indicators": [],
                "details": {
                    "frames_analyzed": 0,
                    "temporal_consistency": 0.0
                }
            }
            
            frame_scores = []
            frame_count = 0
            max_frames = 30  # Анализируем каждый N-й кадр
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames)
            
            prev_frame = None
            temporal_scores = []
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % step == 0:
                    # Анализ отдельного кадра
                    noise_score = self._analyze_noise_patterns(frame)
                    artifact_score = self._detect_ai_artifacts(frame)
                    frame_score = (noise_score + artifact_score) / 2
                    frame_scores.append(frame_score)
                    
                    # Анализ темпоральной согласованности
                    if prev_frame is not None:
                        temporal_score = self._analyze_temporal_consistency(prev_frame, frame)
                        temporal_scores.append(temporal_score)
                    
                    prev_frame = frame.copy()
                
                frame_count += 1
            
            cap.release()
            
            if frame_scores:
                avg_frame_score = np.mean(frame_scores)
                avg_temporal_score = np.mean(temporal_scores) if temporal_scores else 0.5
                
                # AI-видео часто имеет проблемы с темпоральной согласованностью
                results["ai_probability"] = round(
                    (avg_frame_score * 0.6 + avg_temporal_score * 0.4) * 100, 2
                )
                results["details"]["frames_analyzed"] = len(frame_scores)
                results["details"]["temporal_consistency"] = round(avg_temporal_score * 100, 2)
                
                if avg_frame_score > 0.7:
                    results["indicators"].append("AI-артефакты в кадрах")
                if avg_temporal_score > 0.7:
                    results["indicators"].append("Нарушение темпоральной согласованности")
            
            return results
            
        except Exception as e:
            return {"error": f"Ошибка анализа видео: {str(e)}"}
    
    def detect_audio(self, audio_path: str) -> Dict:
        """Анализирует аудио на наличие признаков AI-генерации"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            results = {
                "type": "audio",
                "ai_probability": 0.0,
                "indicators": [],
                "details": {}
            }
            
            # 1. Анализ спектральных характеристик
            spectral_score = self._analyze_audio_spectrum(y, sr)
            results["details"]["spectral_analysis"] = spectral_score
            
            # 2. Анализ паттернов и артефактов
            pattern_score = self._analyze_audio_patterns(y, sr)
            results["details"]["pattern_analysis"] = pattern_score
            
            # 3. Анализ естественности (для голоса)
            naturalness_score = self._analyze_voice_naturalness(y, sr)
            results["details"]["naturalness"] = naturalness_score
            
            # Расчет итоговой вероятности
            total_score = (
                spectral_score * 0.35 +
                pattern_score * 0.35 +
                naturalness_score * 0.30
            )
            
            results["ai_probability"] = round(total_score * 100, 2)
            
            if spectral_score > 0.7:
                results["indicators"].append("Аномалии в спектре")
            if pattern_score > 0.7:
                results["indicators"].append("Подозрительные паттерны")
            if naturalness_score > 0.7:
                results["indicators"].append("Неестественное звучание")
            
            return results
            
        except Exception as e:
            return {"error": f"Ошибка анализа аудио: {str(e)}"}
    
    # === Вспомогательные методы для анализа изображений ===
    
    def _analyze_metadata(self, image_path: str) -> float:
        """Анализ метаданных изображения"""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            img = Image.open(image_path)
            exif = img._getexif()
            
            if exif is None or len(exif) < 3:
                # AI-изображения часто не имеют EXIF данных
                return 0.8
            
            # Проверка на типичные поля камеры
            camera_fields = ['Make', 'Model', 'DateTime', 'Software']
            has_camera_info = any(
                TAGS.get(tag, tag) in camera_fields 
                for tag in exif.keys()
            )
            
            return 0.2 if has_camera_info else 0.7
            
        except:
            return 0.5
    
    def _analyze_noise_patterns(self, img: np.ndarray) -> float:
        """Анализ паттернов шума"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Высокочастотная составляющая (шум)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_std = np.std(laplacian)
        
        # AI-изображения часто имеют неестественно низкий или высокий уровень шума
        natural_noise_range = (5, 25)
        
        if noise_std < natural_noise_range[0]:
            return 0.7  # Слишком "чистое"
        elif noise_std > natural_noise_range[1] * 2:
            return 0.6  # Слишком зашумленное
        else:
            return 0.3
    
    def _analyze_symmetry(self, img: np.ndarray) -> float:
        """Анализ симметрии и повторяющихся паттернов"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Сравнение левой и правой половин
        h, w = gray.shape
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        
        min_width = min(left.shape[1], right.shape[1])
        left = left[:, :min_width]
        right = right[:, :min_width]
        
        similarity = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        
        # AI часто создает слишком симметричные изображения
        if similarity > 0.85:
            return 0.7
        else:
            return 0.3
    
    def _analyze_frequency(self, img: np.ndarray) -> float:
        """Анализ частотных характеристик через FFT"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Анализ распределения частот
        h, w = magnitude.shape
        center_region = magnitude[h//4:3*h//4, w//4:3*w//4]
        edge_region = magnitude.copy()
        edge_region[h//4:3*h//4, w//4:3*w//4] = 0
        
        center_energy = np.sum(center_region)
        edge_energy = np.sum(edge_region)
        
        ratio = center_energy / (edge_energy + 1)
        
        # AI-изображения часто имеют аномальное распределение частот
        if ratio > 100 or ratio < 10:
            return 0.7
        else:
            return 0.3
    
    def _detect_ai_artifacts(self, img: np.ndarray) -> float:
        """Детектирование типичных AI-артефактов"""
        score = 0.0
        count = 0
        
        # 1. Проверка на "checkerboard artifacts" (шахматные артефакты)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Вычисление второй производной
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        checker_variance = np.var(laplacian)
        
        if checker_variance > 1000:
            score += 0.8
        count += 1
        
        # 2. Проверка на странности в краях объектов
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.05 or edge_density > 0.3:
            score += 0.6
        count += 1
        
        # 3. Проверка цветовых аномалий
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            
            # AI часто создает неестественно насыщенные цвета
            if np.mean(saturation) > 180:
                score += 0.7
            count += 1
        
        return score / count if count > 0 else 0.5
    
    def _analyze_temporal_consistency(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Анализ согласованности между кадрами"""
        # Оптический поток
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Анализ величины и согласованности потока
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # AI-видео часто имеет резкие изменения или неестественно плавные переходы
        mean_flow = np.mean(magnitude)
        std_flow = np.std(magnitude)
        
        if mean_flow > 20 or (mean_flow < 2 and std_flow < 1):
            return 0.7
        else:
            return 0.3
    
    # === Вспомогательные методы для анализа аудио ===
    
    def _analyze_audio_spectrum(self, y: np.ndarray, sr: int) -> float:
        """Анализ спектральных характеристик"""
        # Спектрограмма
        spec = np.abs(librosa.stft(y))
        
        # Спектральный центроид
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # AI-голос часто имеет неестественное распределение частот
        mean_centroid = np.mean(spectral_centroids)
        
        if mean_centroid < 1000 or mean_centroid > 4000:
            return 0.7
        else:
            return 0.3
    
    def _analyze_audio_patterns(self, y: np.ndarray, sr: int) -> float:
        """Анализ паттернов в аудио"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # AI-аудио часто имеет слишком регулярные паттерны
        zcr_std = np.std(zcr)
        
        if zcr_std < 0.01:
            return 0.7
        else:
            return 0.3
    
    def _analyze_voice_naturalness(self, y: np.ndarray, sr: int) -> float:
        """Анализ естественности голоса"""
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Анализ вариативности
        mfcc_var = np.var(mfccs, axis=1)
        
        # Синтезированный голос часто имеет меньшую вариативность
        mean_var = np.mean(mfcc_var)
        
        if mean_var < 10:
            return 0.7
        else:
            return 0.3


def format_result(result: Dict) -> str:
    """Форматирование результата для отображения"""
    if "error" in result:
        return f"❌ {result['error']}"
    
    ai_prob = result["ai_probability"]
    content_type = result["type"]
    
    # Определение вердикта
    if ai_prob >= 70:
        verdict = "🤖 Вероятно создано AI"
        emoji = "🔴"
    elif ai_prob >= 40:
        verdict = "⚠️ Возможно создано AI"
        emoji = "🟡"
    else:
        verdict = "✅ Вероятно создано человеком"
        emoji = "🟢"
    
    output = f"{emoji} {verdict}\n"
    output += f"📊 Вероятность AI: {ai_prob}%\n\n"
    
    if result["indicators"]:
        output += "🔍 Обнаруженные признаки:\n"
        for indicator in result["indicators"]:
            output += f"  • {indicator}\n"
        output += "\n"
    
    output += "📈 Детальный анализ:\n"
    for key, value in result["details"].items():
        key_ru = {
            "metadata_analysis": "Метаданные",
            "noise_patterns": "Паттерны шума",
            "symmetry_analysis": "Симметрия",
            "frequency_analysis": "Частотный анализ",
            "ai_artifacts": "AI-артефакты",
            "frames_analyzed": "Проанализировано кадров",
            "temporal_consistency": "Темпоральная согласованность",
            "spectral_analysis": "Спектральный анализ",
            "pattern_analysis": "Анализ паттернов",
            "naturalness": "Естественность"
        }.get(key, key)
        
        if isinstance(value, float):
            output += f"  • {key_ru}: {round(value * 100, 1)}%\n"
        else:
            output += f"  • {key_ru}: {value}\n"
    
    return output


if __name__ == "__main__":
    # Тестирование
    detector = AIContentDetector()
    print("AI Content Detector готов к работе!")
