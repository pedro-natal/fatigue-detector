"""
Detector de Fadiga Especializado em Olhos
Usa modelos treinados APENAS com imagens de olhos
"""

import cv2
import numpy as np
import pickle
import os
import json
from datetime import datetime


class EyeFatigueDetector:
    """
    Detector especializado que usa APENAS análise de olhos
    Ignora boca e outras features, foca 100% nos olhos
    """

    def __init__(self):
        print("👁️ Iniciando Detector Especializado em Fadiga")

        # Carrega modelos especializados
        self.carregar_modelos_olhos()

        # Carrega calibração personalizada
        self.carregar_calibracao()

        # Inicializa OpenCV
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        # Histórico para suavização
        self.historico_olhos = []
        self.max_historico = 8  # Reduzido para resposta mais rápida

        # Estatísticas
        self.total_frames = 0
        self.deteccoes_sono = 0

    def carregar_modelos_olhos(self):
        """Carrega modelos especializados em olhos"""
        try:
            with open("models/eye_fatigue_models.pkl", "rb") as f:
                models = pickle.load(f)

            self.eye_model = models["eye_model"]
            self.eye_scaler = models["eye_scaler"]
            self.feature_size = models["feature_size"]
            self.image_size = models["image_size"]
            self.melhor_modelo = models["melhor_modelo"]

            print("✅ Modelos de olhos carregados com sucesso!")
            print(f"🤖 Modelo ativo: {self.melhor_modelo}")
            print(f"📏 Features: {self.feature_size}")

        except FileNotFoundError:
            print("❌ Modelos de olhos não encontrados!")
            print("🔧 Execute o eye_dataset_trainer.py primeiro!")
            exit(1)
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
            exit(1)

    def carregar_calibracao(self):
        """Carrega calibração personalizada ou usa padrões conservadores"""
        arquivos_calibracao = [
            "eye_calibration.json",
            "super_calibracao.json",
            "calibracao_personalizada.json",
        ]

        calibracao_carregada = False

        for arquivo in arquivos_calibracao:
            if os.path.exists(arquivo):
                try:
                    with open(arquivo, "r") as f:
                        config = json.load(f)

                    self.threshold_sono = config.get("threshold_sono", 0.75)
                    print(f"✅ Calibração carregada: {arquivo}")
                    print(f"🎯 Threshold: {self.threshold_sono:.2f}")
                    calibracao_carregada = True
                    break
                except:
                    continue

        if not calibracao_carregada:
            # Threshold conservador para olhos
            self.threshold_sono = 0.75
            print("⚠️ Usando threshold padrão: 0.75")

    def extrair_features_olho_avancadas(self, roi_olho):
        """
        Extrai features avançadas de uma região de olho
        Mesma função usada no treinamento
        """
        if roi_olho.size == 0:
            return np.zeros(self.feature_size)

        try:
            # Redimensiona para tamanho padrão
            gray = cv2.resize(roi_olho, self.image_size)
            h, w = gray.shape

            features = []

            # 1. Features de intensidade
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            min_intensity = np.min(gray)
            max_intensity = np.max(gray)
            features.extend(
                [mean_intensity, std_intensity, min_intensity, max_intensity]
            )

            # 2. Features de textura (regiões)
            regions = [
                gray[: h // 2, : w // 2],  # Superior esquerdo
                gray[: h // 2, w // 2 :],  # Superior direito
                gray[h // 2 :, : w // 2],  # Inferior esquerdo
                gray[h // 2 :, w // 2 :],  # Inferior direito
            ]

            for region in regions:
                if region.size > 0:
                    features.append(np.mean(region))
                else:
                    features.append(0)

            # 3. Features de bordas
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.count_nonzero(edges) / (h * w)
            edge_mean = np.mean(edges)
            features.extend([edge_density, edge_mean])

            # 4. Features de contorno
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            num_contours = len(contours)
            max_contour_area = (
                max([cv2.contourArea(c) for c in contours]) if contours else 0
            )
            avg_contour_area = (
                np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
            )
            features.extend([num_contours, max_contour_area, avg_contour_area])

            # 5. Features de gradiente
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_mean = np.mean(gradient_magnitude)
            grad_std = np.std(gradient_magnitude)
            features.extend([grad_mean, grad_std])

            # 6. Features de forma
            moments = cv2.moments(gray)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                features.extend([cx / w, cy / h])
            else:
                features.extend([0.5, 0.5])

            # Garante tamanho correto
            while len(features) < self.feature_size:
                features.append(0.0)

            return np.array(features[: self.feature_size])

        except Exception as e:
            print(f"⚠️ Erro ao extrair features: {e}")
            return np.zeros(self.feature_size)

    def detectar_fadiga_frame(self, frame):
        """
        Detecta fadiga focando APENAS nos olhos
        """
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return frame, 0.0, "Nenhuma face detectada"

        # Pega a maior face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        # ROI da face
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detecta olhos
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        probabilidades_sono = []

        if len(eyes) > 0:
            # Analisa cada olho detectado
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Máximo 2 olhos
                roi_olho = roi_gray[ey : ey + eh, ex : ex + ew]

                # Extrai features
                features = self.extrair_features_olho_avancadas(roi_olho)

                if np.any(features):
                    # Normaliza e prediz
                    features_scaled = self.eye_scaler.transform([features])
                    prob_sono = self.eye_model.predict_proba(features_scaled)[0][1]
                    probabilidades_sono.append(prob_sono)

                    # Desenha retângulo no olho
                    cor = (
                        (0, 0, 255) if prob_sono > self.threshold_sono else (0, 255, 0)
                    )
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), cor, 2)

                    # Mostra probabilidade no olho
                    cv2.putText(
                        roi_color,
                        f"{prob_sono:.2f}",
                        (ex, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        cor,
                        1,
                    )

        # Calcula probabilidade final
        if probabilidades_sono:
            # Usa a MAIOR probabilidade entre os olhos (mais conservador)
            prob_sono_final = max(probabilidades_sono)
        else:
            prob_sono_final = 0.0

        # Adiciona ao histórico para suavização
        self.historico_olhos.append(prob_sono_final)
        if len(self.historico_olhos) > self.max_historico:
            self.historico_olhos.pop(0)

        # Probabilidade suavizada
        sono_suavizado = np.mean(self.historico_olhos)

        # Determina estado
        if sono_suavizado > self.threshold_sono:
            estado = "🚨 SONO DETECTADO"
            self.deteccoes_sono += 1
            cor_estado = (0, 0, 255)
        else:
            estado = "✅ ALERTA"
            cor_estado = (0, 255, 0)

        # Calcula fadiga (0-100%)
        fadiga_pct = min(100, sono_suavizado * 100)

        # Desenha informações
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Informações na tela
        info_lines = [
            f"Olhos: {len(eyes)} detectados",
            f"Sono: {sono_suavizado:.1%}",
            f"Fadiga: {fadiga_pct:.0f}%",
            f"Estado: {estado}",
            f"Threshold: {self.threshold_sono:.2f}",
            f"Deteccoes: {self.deteccoes_sono}/{self.total_frames}",
        ]

        y_offset = 30
        for i, line in enumerate(info_lines):
            cor = cor_estado if "Estado:" in line else (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cor,
                2,
            )

        return frame, fadiga_pct / 100, estado

    def executar(self):
        """Executa o detector especializado em olhos"""
        print("\n👁️ DETECTOR DE FADIGA INICIADO")
        print("📹 Pressione 'q' para sair")
        print("📹 Pressione 'c' para recalibrar threshold")
        print("📹 Pressione 's' para estatísticas\n")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Erro ao acessar webcam!")
            return

        try:
            inicio = datetime.now()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_processado, fadiga, estado = self.detectar_fadiga_frame(frame)

                # Alertas no terminal
                if fadiga > 0.7:
                    print(f"🚨 ALERTA: {estado} - Fadiga: {fadiga:.1%}")

                cv2.imshow("Detector Especializado em Fadiga", frame_processado)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.calibrar_threshold()
                elif key == ord("s"):
                    self.mostrar_estatisticas(inicio)

        except KeyboardInterrupt:
            print("\n⏹️ Detector interrompido")
        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Estatísticas finais
            duracao = datetime.now() - inicio
            print(f"\n📊 ESTATÍSTICAS FINAIS:")
            print(f"⏱️ Duração: {duracao}")
            print(f"📹 Frames processados: {self.total_frames}")
            print(f"🚨 Detecções de sono: {self.deteccoes_sono}")
            if self.total_frames > 0:
                print(
                    f"📈 Taxa de sono: {self.deteccoes_sono/self.total_frames*100:.1f}%"
                )

    def calibrar_threshold(self):
        """Calibra threshold interativamente"""
        print("\n🎯 CALIBRAÇÃO INTERATIVA")
        print("Digite novo threshold (0.0 - 1.0) ou Enter para manter atual:")

        try:
            entrada = input(f"Threshold atual: {self.threshold_sono:.2f} -> ").strip()
            if entrada:
                novo_threshold = float(entrada)
                if 0.0 <= novo_threshold <= 1.0:
                    self.threshold_sono = novo_threshold
                    print(f"✅ Threshold atualizado: {self.threshold_sono:.2f}")

                    # Salva nova calibração
                    config = {
                        "threshold_sono": self.threshold_sono,
                        "calibration_date": datetime.now().isoformat(),
                        "detector_type": "eye_specialized",
                    }

                    with open("eye_calibration.json", "w") as f:
                        json.dump(config, f, indent=2)

                    print("💾 Calibração salva!")
                else:
                    print("❌ Threshold deve estar entre 0.0 e 1.0")
        except ValueError:
            print("❌ Valor inválido")

    def mostrar_estatisticas(self, inicio):
        """Mostra estatísticas em tempo real"""
        duracao = datetime.now() - inicio

        print(f"\n📊 ESTATÍSTICAS ATUAIS:")
        print(f"⏱️ Tempo rodando: {duracao}")
        print(f"📹 Frames: {self.total_frames}")
        print(f"🚨 Detecções: {self.deteccoes_sono}")
        print(f"🎯 Threshold: {self.threshold_sono:.2f}")
        if self.total_frames > 0:
            print(f"📈 Taxa sono: {self.deteccoes_sono/self.total_frames*100:.1f}%")
        print()


def main():
    """Função principal"""
    detector = EyeFatigueDetector()
    detector.executar()


if __name__ == "__main__":
    main()
