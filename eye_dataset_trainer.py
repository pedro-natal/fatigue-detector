"""
Treinador Especializado em Dataset de Olhos
Treina modelos usando APENAS imagens de olhos para maior precisão
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


class EyeDatasetTrainer:
    """
    Treinador especializado para datasets de olhos
    Foca APENAS na região dos olhos para máxima precisão
    """

    def __init__(self):
        print("Iniciando Treinador de Dataset de Olhos")
        print("Este sistema treina com imagens de olhos!")

        self.models = {}
        self.scalers = {}
        self.training_history = []

        # Configurações otimizadas para olhos
        self.image_size = (64, 32)  # Proporção ideal para olhos
        self.feature_size = 20  # Features específicas para olhos

    def preparar_dataset_olhos(self, dataset_path):
        """
        Prepara dataset focado APENAS em olhos
        Aceita estrutura: dataset/open_eyes/ e dataset/closed_eyes/
        Ou: dataset/alert/ e dataset/drowsy/
        """
        print(f"\n📁 Preparando dataset de olhos: {dataset_path}")

        # Detecta estrutura do dataset
        estruturas_possiveis = [
            (["open_eyes", "closed_eyes"], "open_eyes", "closed_eyes"),
            (["alert", "drowsy"], "alert", "drowsy"),
            (["awake", "sleepy"], "awake", "sleepy"),
            (["normal", "tired"], "normal", "tired"),
        ]

        pasta_alerta = None
        pasta_sono = None

        for pastas, p_alerta, p_sono in estruturas_possiveis:
            if all(os.path.exists(os.path.join(dataset_path, p)) for p in pastas):
                pasta_alerta = p_alerta
                pasta_sono = p_sono
                print(f"Estrutura detectada: {pasta_alerta} / {pasta_sono}")
                break

        if not pasta_alerta:
            print("❌ Estrutura de dataset não reconhecida!")
            print("\nEstruturas suportadas:")
            print("   • dataset/open_eyes/ e dataset/closed_eyes/")
            print("   • dataset/alert/ e dataset/drowsy/")
            print("   • dataset/awake/ e dataset/sleepy/")
            print("   • dataset/normal/ e dataset/tired/")
            return None, None, None, None

        # Processa imagens
        features = []
        labels = []

        # Imagens de alerta (olhos abertos)
        pasta_alerta_path = os.path.join(dataset_path, pasta_alerta)
        alerta_count = 0

        print(f"\n📂 Processando imagens de alerta: {pasta_alerta}")
        for filename in os.listdir(pasta_alerta_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(pasta_alerta_path, filename)
                feature = self.extrair_features_olho_avancadas(img_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(0)  # 0 = alerta
                    alerta_count += 1

                    if alerta_count % 100 == 0:
                        print(f"   Processadas: {alerta_count}")

        # Imagens de sono (olhos fechados)
        pasta_sono_path = os.path.join(dataset_path, pasta_sono)
        sono_count = 0

        print(f"\n📂 Processando imagens de sono: {pasta_sono}")
        for filename in os.listdir(pasta_sono_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(pasta_sono_path, filename)
                feature = self.extrair_features_olho_avancadas(img_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(1)  # 1 = sono
                    sono_count += 1

                    if sono_count % 100 == 0:
                        print(f"   Processadas: {sono_count}")

        print(f"\nDataset preparado:")
        print(f"   Imagens de alerta: {alerta_count}")
        print(f"   Imagens de sono: {sono_count}")
        print(f"   📊 Total: {len(features)} imagens")

        return np.array(features), np.array(labels), alerta_count, sono_count

    def extrair_features_olho_avancadas(self, img_path):
        """
        Extrai features avançadas específicas para análise de olhos
        """
        try:
            # Carrega imagem
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Converte para grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Redimensiona para tamanho padrão
            gray = cv2.resize(gray, self.image_size)
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

            # 2. Features de textura (LBP simplificado)
            # Divide imagem em regiões
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

            # 3. Features de bordas (Canny)
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

            # 6. Features de forma (momentos)
            moments = cv2.moments(gray)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                features.extend([cx / w, cy / h])  # Normalizado
            else:
                features.extend([0.5, 0.5])

            # Garante que temos exatamente o número correto de features
            while len(features) < self.feature_size:
                features.append(0.0)

            return np.array(features[: self.feature_size])

        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
            return None

    def treinar_modelos(self, X, y):
        """
        Treina múltiplos modelos otimizados para detecção de olhos
        """
        print("\n🤖 Iniciando treinamento dos modelos...")

        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Dados de treinamento: {len(X_train)}")
        print(f"📊 Dados de teste: {len(X_test)}")

        # Normalização
        self.scalers["eye_scaler"] = StandardScaler()
        X_train_scaled = self.scalers["eye_scaler"].fit_transform(X_train)
        X_test_scaled = self.scalers["eye_scaler"].transform(X_test)

        # Modelos otimizados para olhos
        modelos = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "SVM": SVC(
                C=10, gamma="scale", kernel="rbf", probability=True, random_state=42
            ),
        }

        resultados = {}

        for nome, modelo in modelos.items():
            print(f"\n🔄 Treinando {nome}...")

            # Treina modelo
            modelo.fit(X_train_scaled, y_train)

            # Predições
            y_pred = modelo.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Salva modelo
            self.models[f"eye_model_{nome.lower()}"] = modelo

            # Salva resultados
            resultados[nome] = {
                "accuracy": accuracy,
                "predictions": y_pred,
                "test_labels": y_test,
            }

            print(f"{nome} - Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Relatório de classificação:")
            print(
                classification_report(y_test, y_pred, target_names=["Alerta", "Sono"])
            )

        # Seleciona melhor modelo
        melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]["accuracy"])
        self.melhor_modelo = melhor_modelo.lower()

        print(f"\n🏆 Melhor modelo: {melhor_modelo}")
        print(f"Acurácia final: {resultados[melhor_modelo]['accuracy']*100:.2f}%")

        return resultados

    def salvar_modelos(self):
        """Salva modelos treinados"""
        os.makedirs("models", exist_ok=True)

        # Salva modelos principais
        modelo_data = {
            "eye_model": self.models[f"eye_model_{self.melhor_modelo}"],
            "eye_scaler": self.scalers["eye_scaler"],
            "melhor_modelo": self.melhor_modelo,
            "feature_size": self.feature_size,
            "image_size": self.image_size,
            "training_date": datetime.now().isoformat(),
            "dataset_type": "eyes_only",
        }

        with open("models/eye_fatigue_models.pkl", "wb") as f:
            pickle.dump(modelo_data, f)

        print("Modelos salvos em: models/eye_fatigue_models.pkl")

        # Salva configuração
        config = {
            "melhor_modelo": self.melhor_modelo,
            "feature_size": self.feature_size,
            "image_size": self.image_size,
            "dataset_type": "eyes_only",
            "training_date": datetime.now().isoformat(),
        }

        with open("models/eye_training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("Configuração salva em: models/eye_training_config.json")

    def visualizar_resultados(self, resultados):
        """Visualiza resultados do treinamento"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico de acurácia
        modelos = list(resultados.keys())
        acuracias = [resultados[m]["accuracy"] for m in modelos]

        axes[0].bar(modelos, acuracias, color=["#2ecc71", "#3498db"])
        axes[0].set_title("Acurácia dos Modelos (Dataset de Olhos)")
        axes[0].set_ylabel("Acurácia")
        axes[0].set_ylim(0, 1)

        for i, acc in enumerate(acuracias):
            axes[0].text(i, acc + 0.01, f"{acc:.3f}", ha="center")

        # Matriz de confusão do melhor modelo
        melhor = max(resultados.keys(), key=lambda k: resultados[k]["accuracy"])
        y_test = resultados[melhor]["test_labels"]
        y_pred = resultados[melhor]["predictions"]

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)

        im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        axes[1].set_title(f"Matriz de Confusão - {melhor}")

        classes = ["Alerta", "Sono"]
        tick_marks = np.arange(len(classes))
        axes[1].set_xticks(tick_marks)
        axes[1].set_yticks(tick_marks)
        axes[1].set_xticklabels(classes)
        axes[1].set_yticklabels(classes)

        # Adiciona números na matriz
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            axes[1].text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()

        # Salva gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            f"eye_training_results_{timestamp}.png", dpi=300, bbox_inches="tight"
        )
        print(f"📊 Gráficos salvos: eye_training_results_{timestamp}.png")

        plt.show()


def main():
    """Função principal"""
    print("TREINADOR DE DATASET DE OLHOS")
    print("=" * 50)

    trainer = EyeDatasetTrainer()

    while True:
        print("\nOPÇÕES:")
        print("1. Treinar com novo dataset de olhos")
        print("2. Verificar modelos existentes")
        print("3. Sair")

        opcao = input("\nEscolha uma opção (1-3): ").strip()

        if opcao == "1":
            print("\n📁 CONFIGURAÇÃO DO DATASET")
            print("Estruturas suportadas:")
            print("   • dataset/open_eyes/ e dataset/closed_eyes/")
            print("   • dataset/alert/ e dataset/drowsy/")
            print("   • dataset/awake/ e dataset/sleepy/")

            dataset_path = input("\nCaminho para o dataset de olhos: ").strip()

            if not os.path.exists(dataset_path):
                print("❌ Caminho não encontrado!")
                continue

            # Prepara dataset
            X, y, alerta_count, sono_count = trainer.preparar_dataset_olhos(
                dataset_path
            )

            if X is None:
                continue

            # Treina modelos
            resultados = trainer.treinar_modelos(X, y)

            # Salva modelos
            trainer.salvar_modelos()

            # Visualiza resultados
            trainer.visualizar_resultados(resultados)

            print("\n🎉 TREINAMENTO CONCLUÍDO!")
            print("Modelos salvos e prontos para uso")
            print("Use o Detector de Olhos para testar!")

        elif opcao == "2":
            if os.path.exists("models/eye_fatigue_models.pkl"):
                print("Modelos de olhos encontrados!")

                if os.path.exists("models/eye_training_config.json"):
                    with open("models/eye_training_config.json", "r") as f:
                        config = json.load(f)

                    print(f"📊 Modelo: {config['melhor_modelo']}")
                    print(f"📅 Treinado em: {config['training_date']}")
                    print(f"Tipo: {config['dataset_type']}")
            else:
                print("❌ Nenhum modelo de olhos encontrado!")
                print("Execute o treinamento primeiro.")

        elif opcao == "3":
            print("👋 Saindo...")
            break

        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    main()
