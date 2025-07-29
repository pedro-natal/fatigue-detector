"""
Sistema de Detecção de Fadiga por Olhos
Interface principal do sistema
"""

import tkinter as tk
from tkinter import messagebox
import os
import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Verifica dependências instaladas"""
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
    """Retorna o executável Python do projeto"""
    venv_python = r".\.venv\Scripts\python.exe"
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def run_eye_dataset_organizer():
    """Executa o organizador de dataset"""
    try:
        python_path = get_python_executable()
        script_path = "eye_dataset_organizer.py"

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "Organizador de Dataset",
                "EXTRATOR DE IMAGENS DE OLHOS\n\n"
                + "Esta ferramenta vai:\n"
                + "• Extrair regiões de olhos do dataset\n"
                + "• Organizar em pastas alert/drowsy\n"
                + "• Redimensionar para 64x32 pixels\n"
                + "• Preparar para treinamento\n\n"
                + "Estruturas suportadas:\n"
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
                    "Organizador iniciado!\n\n"
                    + "PASSOS:\n"
                    + "1. Informe o caminho do dataset original\n"
                    + "2. Informe onde salvar as imagens extraídas\n"
                    + "3. Aguarde a extração\n\n"
                    + "Depois use o Treinador!",
                )
        else:
            messagebox.showerror("Erro", "Arquivo do organizador não encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar organizador: {e}")


def run_eye_trainer():
    """Executa o treinador de modelos"""
    try:
        python_path = get_python_executable()
        script_path = "eye_dataset_trainer.py"

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "Treinador de Modelos",
                "TREINAMENTO DE MODELOS\n\n"
                + "• Usa imagens de olhos\n"
                + "• Extrai características avançadas\n"
                + "• Treina modelos RandomForest e SVM\n"
                + "• Reduz falsos positivos\n\n"
                + "Você tem um dataset preparado?",
            )

            if result:
                subprocess.Popen(
                    [python_path, script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

                messagebox.showinfo(
                    "Treinador Iniciado",
                    "Treinador iniciado!\n\n"
                    + "PROCESSO:\n"
                    + "1. Informe o caminho do dataset\n"
                    + "2. Aguarde extração de características\n"
                    + "3. Aguarde treinamento\n"
                    + "4. Veja os resultados\n\n"
                    + "Modelos salvos automaticamente!",
                )
        else:
            messagebox.showerror("Erro", "Arquivo do treinador não encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao executar treinador: {e}")


def run_eye_detector():
    """Executa o detector de fadiga"""
    try:
        python_path = get_python_executable()
        script_path = "eye_fatigue_detector.py"

        if os.path.exists(script_path):
            result = messagebox.askyesno(
                "Detector de Fadiga",
                "DETECTOR DE FADIGA\n\n"
                + "• Usa modelos treinados\n"
                + "• Detecta posição da cabeça\n"
                + "• Calibração personalizada\n\n"
                + "REQUISITO: Modelos treinados\n\n"
                + "Continuar?",
            )

            if result:
                subprocess.Popen(
                    [python_path, script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

                messagebox.showinfo(
                    "Detector Iniciado",
                    "Detector iniciado!\n\n"
                    + "CONTROLES:\n"
                    + "• 'q' = Sair\n"
                    + "• 'c' = Calibrar threshold\n"
                    + "• 's' = Ver estatísticas\n"
                    + "• 'r' = Resetar posição da cabeça\n\n"
                    + "ALERTAS:\n"
                    + "• Olhos fechados por 3+ segundos\n"
                    + "• Cabeça baixa por 5+ segundos\n"
                    + "• Incluem alertas sonoros",
                )
        else:
            messagebox.showerror("Erro", "Arquivo não encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro: {e}")


def check_models():
    """Verifica status dos modelos"""
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
    report = "STATUS DO SISTEMA\n\n"

    # Status dos modelos
    if os.path.exists(eye_models_path):
        report += "Modelos especializados: ENCONTRADOS\n"
        report += "Sistema pronto\n\n"
        model_status = "ready"
    elif os.path.exists(fatigue_models_path):
        report += "Modelos básicos: ENCONTRADOS\n"
        report += "Recomenda-se treinar modelos especializados\n\n"
        model_status = "basic"
    else:
        report += "Nenhum modelo encontrado\n"
        report += "Execute o treinamento primeiro\n\n"
        model_status = "missing"

    # Status dos arquivos essenciais
    report += "ARQUIVOS ESSENCIAIS:\n"
    all_present = True
    for name, filepath in essential_files.items():
        if os.path.exists(filepath):
            report += f"{name}: OK\n"
        else:
            report += f"{name}: FALTANDO\n"
            all_present = False

    # Dataset status
    if os.path.exists("eye_dataset"):
        try:
            alert_count = len(os.listdir("eye_dataset/alert"))
            drowsy_count = len(os.listdir("eye_dataset/drowsy"))
            report += f"\nDATASET:\n"
            report += f"Olhos alerta: {alert_count}\n"
            report += f"Olhos sonolência: {drowsy_count}\n"
        except:
            report += f"\nDATASET: Presente mas inacessível\n"
    else:
        report += f"\nDATASET: Não encontrado\n"

    # Recomendações
    if model_status == "ready" and all_present:
        report += "\nSISTEMA COMPLETO E FUNCIONAL!"
        messagebox.showinfo("Status do Sistema", report)
    elif model_status == "missing":
        report += "\nAÇÕES NECESSÁRIAS:\n"
        report += "1. Organizar Dataset\n"
        report += "2. Treinar Modelos\n"
        report += "3. Executar Detector"
        messagebox.showwarning("Sistema Incompleto", report)
    else:
        messagebox.showinfo("Status do Sistema", report)


def create_gui():
    """Cria a interface gráfica"""
    root = tk.Tk()
    root.title("Sistema de Detecção de Fadiga")
    root.geometry("700x800")
    root.configure(bg="#2c3e50")
    root.resizable(True, True)

    # Header
    header_frame = tk.Frame(root, bg="#34495e", height=70)
    header_frame.pack(fill="x", padx=20, pady=5)
    header_frame.pack_propagate(False)

    title_label = tk.Label(
        header_frame,
        text="SISTEMA DE DETECÇÃO DE FADIGA",
        font=("Arial", 14, "bold"),
        fg="white",
        bg="#34495e",
    )
    title_label.pack(expand=True)

    # Frame principal
    main_frame = tk.Frame(root, bg="#2c3e50")
    main_frame.pack(expand=True, fill="both", padx=40, pady=10)

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
        "pady": 8,
        "width": 40,
    }

    # Detector principal
    main_section = tk.Label(
        main_frame,
        text="DETECTOR PRINCIPAL",
        font=("Arial", 12, "bold"),
        fg="#e67e22",
        bg="#2c3e50",
    )
    main_section.pack(pady=(15, 8))

    eye_detector_btn = tk.Button(
        main_frame,
        text="DETECTOR DE FADIGA",
        bg="#e67e22",
        command=run_eye_detector,
        **btn_style,
    )
    eye_detector_btn.pack(pady=5)

    # Preparação de dados
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
    organizer_btn.pack(pady=5)

    trainer_btn = tk.Button(
        main_frame,
        text="TREINAR MODELOS",
        bg="#2980b9",
        command=run_eye_trainer,
        **btn_style,
    )
    trainer_btn.pack(pady=5)

    # Status do sistema
    status_section = tk.Label(
        main_frame,
        text="STATUS DO SISTEMA",
        font=("Arial", 12, "bold"),
        fg="#95a5a6",
        bg="#2c3e50",
    )
    status_section.pack(pady=(15, 8))

    models_btn = tk.Button(
        main_frame,
        text="VERIFICAR STATUS",
        bg="#f39c12",
        command=check_models,
        **btn_style,
    )
    models_btn.pack(pady=5)

    # Instruções
    instructions_section = tk.Label(
        main_frame,
        text="FLUXO RECOMENDADO",
        font=("Arial", 11, "bold"),
        fg="#3498db",
        bg="#2c3e50",
    )
    instructions_section.pack(pady=(15, 8))

    instructions_text = """
1. Organize seu dataset usando "Organizar Dataset"
2. Treine os modelos com "Treinar Modelos"  
3. Use "Detector de Fadiga" para detecção em tempo real

Para verificar status: Use "Verificar Status" """

    instructions_label = tk.Label(
        main_frame,
        text=instructions_text,
        font=("Arial", 9),
        fg="#bdc3c7",
        bg="#2c3e50",
        justify="center",
    )
    instructions_label.pack(pady=10)

    footer_label = tk.Label(
        main_frame,
        text="Detector de Fadiga • OpenCV + Scikit-learn",
        font=("Arial", 8),
        fg="#95a5a6",
        bg="#2c3e50",
    )
    footer_label.pack(pady=15)

    # Verifica dependências
    all_ok, missing_deps, available_deps = check_requirements()

    if not all_ok:
        deps_text = "\n".join(missing_deps)
        messagebox.showwarning(
            "Dependências Ausentes",
            f"Dependências faltando:\n{deps_text}\n\n"
            + "Execute no terminal:\n"
            + f"pip install {' '.join(missing_deps)}",
        )
    elif available_deps:
        print("Dependências instaladas:")
        for name, version in available_deps.items():
            print(f"  {name}: {version}")

    return root


def main():
    """Função principal"""
    print("Iniciando interface...")

    try:
        root = create_gui()
        root.mainloop()
    except Exception as e:
        print(f"Erro ao iniciar interface: {e}")
        messagebox.showerror("Erro", f"Erro ao iniciar interface: {e}")


if __name__ == "__main__":
    main()
