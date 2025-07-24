"""
Sistema de Detecção de Fadiga por Olhos - Interface Principal
Versão refatorada e otimizada focada em análise de olhos
"""

import tkinter as tk
from tkinter import messagebox
import os
import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Verifica se os requisitos estão instalados e mostra versões"""
    missing_deps = []
    available_deps = {}

    try:
        import cv2

        available_deps["OpenCV"] = cv2.__version__
    except ImportError:
        missing_deps.append("opencv-python")

    try:
        import numpy as np

        available_deps["NumPy"] = np.__version__
    except ImportError:
        missing_deps.append("numpy")

    try:
        import sklearn

        available_deps["Scikit-learn"] = sklearn.__version__
    except ImportError:
        missing_deps.append("scikit-learn")

    try:
        import matplotlib

        available_deps["Matplotlib"] = matplotlib.__version__
    except ImportError:
        missing_deps.append("matplotlib")

    if missing_deps:
        return False, missing_deps, available_deps
    return True, [], available_deps


def get_python_executable():
    """Retorna o executável Python correto (venv ou sistema)"""
    venv_python = r".\.venv\Scripts\python.exe"
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def run_eye_dataset_organizer():
    """Executa o organizador de dataset de olhos"""
    try:
        python_path = get_python_executable()
        script_path = "eye_dataset_organizer.py"

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "Extrator de Dataset de Olhos",
                "👁️ EXTRATOR DE DATASE DE OLHOS\n\n"
                + "Esta ferramenta vai:\n"
                + "✅ Extrair APENAS regiões de olhos do seu dataset\n"
                + "✅ Organizar em alert/drowsy automaticamente\n"
                + "✅ Redimensionar para tamanho ideal (64x32)\n"
                + "✅ Preparar para treinamento especializado\n\n"
                + "📂 Estruturas suportadas:\n"
                + "• drowsy/non_drowsy\n"
                + "• alert/drowsy\n"
                + "• open_eyes/closed_eyes\n\n"
                + "Continuar?",
            )

            if result:
                subprocess.Popen(
                    [python_path, script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

                messagebox.showinfo(
                    "Organizador Iniciado",
                    "👁️ Organizador de Dataset iniciado!\n\n"
                    + "📋 PASSOS:\n"
                    + "1️⃣ Informe o caminho do dataset original\n"
                    + "2️⃣ Informe onde salvar os olhos extraídos\n"
                    + "3️⃣ Aguarde a extração automática\n\n"
                    + "✅ Depois use o Treinador de Olhos!",
                )
        else:
            messagebox.showerror("Erro", "Arquivo do organizador não encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar organizador: {e}")


def run_eye_trainer():
    """Executa o treinador de dataset de olhos"""
    try:
        python_path = get_python_executable()
        script_path = "eye_dataset_trainer.py"

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "Treinador de Dataset de Olhos",
                "🤖 TREINADOR ESPECIALIZADO EM OLHOS\n\n"
                + "✅ Usa APENAS imagens de olhos\n"
                + "✅ Features avançadas para análise de olhos\n"
                + "✅ Modelos otimizados (RandomForest + SVM)\n"
                + "✅ Elimina falsos positivos de faces\n\n"
                + "📊 Espere acurácia > 90%!\n\n"
                + "Você tem um dataset de olhos preparado?\n"
                + "(Use o Extrator se precisar extrair olhos)",
            )

            if result:
                subprocess.Popen(
                    [python_path, script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

                messagebox.showinfo(
                    "Treinador Iniciado",
                    "🤖 Treinador de Olhos iniciado!\n\n"
                    + "📋 PROCESSO:\n"
                    + "1️⃣ Informe o caminho do dataset de olhos\n"
                    + "2️⃣ Aguarde extração de features\n"
                    + "3️⃣ Aguarde treinamento dos modelos\n"
                    + "4️⃣ Veja os resultados e gráficos\n\n"
                    + "✅ Modelos salvos automaticamente!",
                )
        else:
            messagebox.showerror("Erro", "Arquivo do treinador não encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar treinador: {e}")


def run_eye_detector():
    """Executa o detector especializado em olhos"""
    try:
        python_path = get_python_executable()
        script_path = "eye_fatigue_detector.py"

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "Detector Especializado em Fadiga",
                "👁️ DETECTOR ESPECIALIZADO EM FADIGA\n\n"
                + "✅ Usa modelos treinados com dataset de olhos\n"
                + "✅ Detecta automaticamente variações de posição de cabeça\n"
                + "✅ Calibração personalizada\n"
                + "⚠️ REQUISITO: Modelos de olhos treinados\n"
                + "(Treine o modelo de olhos primeiro)\n\n"
                + "Continuar?",
            )

            if result:
                subprocess.Popen(
                    [python_path, script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

                messagebox.showinfo(
                    "Detector de Fadiga Iniciado",
                    "👁️ Detector Especializado iniciado!\n\n"
                    + "📋 CONTROLES:\n"
                    + "• 'q' = Sair\n"
                    + "• 'c' = Calibrar threshold\n"
                    + "• 's' = Ver estatísticas\n"
                    + "• 'r' = Resetar posição da cabeça\n\n"
                    + "🚨 ALERTAS AUTOMÁTICOS:\n"
                    + "• Olhos fechados por 3+ segundos\n"
                    + "• Cabeça baixa por 5+ segundos\n"
                    + "• Incluem alertas sonoros",
                )
        else:
            messagebox.showerror("Erro", "Arquivo do detector de olhos não encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar detector de olhos: {e}")


def check_models():
    """Verifica status dos modelos e componentes do sistema"""
    # Verifica modelos
    eye_models_path = "models/eye_fatigue_models.pkl"
    fatigue_models_path = "models/fatigue_models.pkl"

    # Verifica arquivos essenciais
    essential_files = {
        "Haarcascade Face": "haarcascade_frontalface_default.xml",
        "Haarcascade Eye": "haarcascade_eye.xml",
        "Calibração": "eye_calibration.json",
    }

    # Constrói relatório
    report = "📊 STATUS DO SISTEMA\n\n"

    # Status dos modelos
    if os.path.exists(eye_models_path):
        report += "✅ Modelos de olhos especializados: ENCONTRADOS\n"
        report += "🎯 Sistema pronto para detecção de alta precisão\n\n"
        model_status = "ready"
    elif os.path.exists(fatigue_models_path):
        report += "⚠️ Apenas modelos básicos: ENCONTRADOS\n"
        report += "💡 Recomenda-se treinar modelos especializados\n\n"
        model_status = "basic"
    else:
        report += "❌ Nenhum modelo encontrado\n"
        report += "🔧 Execute o treinamento primeiro\n\n"
        model_status = "missing"

    # Status dos arquivos essenciais
    report += "📁 ARQUIVOS ESSENCIAIS:\n"
    all_present = True
    for name, filepath in essential_files.items():
        if os.path.exists(filepath):
            report += f"✅ {name}: OK\n"
        else:
            report += f"❌ {name}: FALTANDO\n"
            all_present = False

    # Dataset status
    if os.path.exists("eye_dataset"):
        try:
            alert_count = len(os.listdir("eye_dataset/alert"))
            drowsy_count = len(os.listdir("eye_dataset/drowsy"))
            report += f"\n📂 DATASET:\n"
            report += f"👁️ Olhos alerta: {alert_count}\n"
            report += f"😴 Olhos sonolência: {drowsy_count}\n"
        except:
            report += f"\n📂 DATASET: Presente mas inacessível\n"
    else:
        report += f"\n📂 DATASET: Não encontrado\n"

    # Recomendações
    if model_status == "ready" and all_present:
        report += "\n🎉 SISTEMA COMPLETO E FUNCIONAL!"
        messagebox.showinfo("Status do Sistema", report)
    elif model_status == "missing":
        report += "\n📋 AÇÕES NECESSÁRIAS:\n"
        report += "1. Organizar Dataset de Olhos\n"
        report += "2. Treinar Modelos de Olhos\n"
        report += "3. Executar Detector"
        messagebox.showwarning("Sistema Incompleto", report)
    else:
        messagebox.showinfo("Status do Sistema", report)


def create_gui():
    """Cria a interface gráfica principal"""
    root = tk.Tk()
    root.title("Sistema de Detecção de Fadiga")
    root.geometry("700x800")  # Aumentei a altura
    root.configure(bg="#2c3e50")
    root.resizable(True, True)

    # Header centralizado
    header_frame = tk.Frame(root, bg="#34495e", height=70)  # Reduzi altura
    header_frame.pack(fill="x", padx=20, pady=5)  # Reduzi pady
    header_frame.pack_propagate(False)

    title_label = tk.Label(
        header_frame,
        text="SISTEMA DE DETECÇÃO DE FADIGA",
        font=("Arial", 14, "bold"),
        fg="white",
        bg="#34495e",
    )
    title_label.pack(expand=True)

    # Frame principal centralizado
    main_frame = tk.Frame(root, bg="#2c3e50")
    main_frame.pack(expand=True, fill="both", padx=40, pady=10)  # Reduzi pady

    info_label = tk.Label(
        main_frame,
        font=("Arial", 9),
        fg="#ecf0f1",
        bg="#2c3e50",
        justify="center",
    )
    info_label.pack(pady=10)  # Reduzi pady

    # Estilo dos botões
    btn_style = {
        "font": ("Arial", 11, "bold"),
        "fg": "white",
        "relief": "raised",
        "bd": 2,
        "pady": 8,  # Reduzi pady
        "width": 40,
    }

    # === DETECTOR PRINCIPAL ===
    main_section = tk.Label(
        main_frame,
        text="DETECTOR PRINCIPAL",
        font=("Arial", 12, "bold"),
        fg="#e67e22",
        bg="#2c3e50",
    )
    main_section.pack(pady=(15, 8))  # Reduzi pady

    # Detector de Olhos
    eye_detector_btn = tk.Button(
        main_frame,
        text="👁️ DETECTOR DE FADIGA",
        bg="#e67e22",
        command=run_eye_detector,
        **btn_style,
    )
    eye_detector_btn.pack(pady=5)  # Reduzi pady

    # === PREPARAÇÃO DE DADOS ===
    data_section = tk.Label(
        main_frame,
        text="PREPARAÇÃO DE DADOS",
        font=("Arial", 12, "bold"),
        fg="#8e44ad",
        bg="#2c3e50",
    )
    data_section.pack(pady=(15, 8))  # Reduzi pady

    # Organizador de Dataset
    organizer_btn = tk.Button(
        main_frame,
        text="📁 EXTRATOR DE OLHOS PARA DATASET",
        bg="#8e44ad",
        command=run_eye_dataset_organizer,
        **btn_style,
    )
    organizer_btn.pack(pady=5)  # Reduzi pady

    # Treinador de Modelos
    trainer_btn = tk.Button(
        main_frame,
        text="🤖 TREINAR MODELOS DE OLHOS",
        bg="#2980b9",
        command=run_eye_trainer,
        **btn_style,
    )
    trainer_btn.pack(pady=5)  # Reduzi pady

    # === STATUS DO SISTEMA ===
    status_section = tk.Label(
        main_frame,
        text="STATUS DO SISTEMA",
        font=("Arial", 12, "bold"),
        fg="#95a5a6",
        bg="#2c3e50",
    )
    status_section.pack(pady=(15, 8))  # Reduzi pady

    # Verificar modelos
    models_btn = tk.Button(
        main_frame,
        text="🔍 VERIFICAR STATUS DOS MODELOS",
        bg="#f39c12",
        command=check_models,
        **btn_style,
    )
    models_btn.pack(pady=5)  # Reduzi pady

    # === INSTRUÇÕES ===
    instructions_section = tk.Label(
        main_frame,
        text="📋 FLUXO RECOMENDADO",
        font=("Arial", 11, "bold"),
        fg="#3498db",
        bg="#2c3e50",
    )
    instructions_section.pack(pady=(15, 8))  # Reduzi pady

    instructions_text = """
1. Organize seu dataset de olhos usando "Organizar Dataset"
2. Treine os modelos com "Treinar Modelos de Olhos"  
3. Use "Detector de Fadiga" para detecção avançada em tempo real

Para status: Use "Verificar Status dos Modelos" """

    instructions_label = tk.Label(
        main_frame,
        text=instructions_text,
        font=("Arial", 9),
        fg="#bdc3c7",
        bg="#2c3e50",
        justify="center",
    )
    instructions_label.pack(pady=10)  # Reduzi pady

    # Footer
    footer_label = tk.Label(
        main_frame,
        text="Detector de Fadiga • OpenCV + Scikit-learn • Dataset Personalizado",
        font=("Arial", 8),
        fg="#95a5a6",
        bg="#2c3e50",
    )
    footer_label.pack(pady=15)  # Reduzi pady

    # Verifica requisitos na inicialização
    all_ok, missing_deps, available_deps = check_requirements()

    if not all_ok:
        deps_text = "\n".join(missing_deps)
        messagebox.showwarning(
            "Dependências Ausentes",
            f"⚠️ Dependências faltando:\n{deps_text}\n\n"
            + "Execute no terminal:\n"
            + f"pip install {' '.join(missing_deps)}",
        )
    elif available_deps:
        # Mostra versões disponíveis no console
        print("📦 Dependências instaladas:")
        for name, version in available_deps.items():
            print(f"  ✅ {name}: {version}")

    return root


def main():
    """Função principal"""
    print("🚀 Iniciando Central de Controle...")

    try:
        root = create_gui()
        root.mainloop()
    except Exception as e:
        print(f"❌ Erro ao iniciar interface: {e}")
        messagebox.showerror("Erro", f"Erro ao iniciar interface: {e}")


if __name__ == "__main__":
    main()
