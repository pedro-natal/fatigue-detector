# 👁️ Sistema de Detecção de Fadiga por Olhos

Sistema especializado em detecção de fadiga através da análise de regiões oculares usando Machine Learning.

## 🚀 Características

- **Detecção Especializada**: Foca apenas nas regiões dos olhos para máxima precisão
- **Interface Gráfica**: GUI intuitiva para todas as funcionalidades
- **Dataset Personalizado**: Organize e treine com seus próprios dados
- **Modelos Otimizados**: RandomForest e SVM especializados em olhos

## 📋 Pré-requisitos

- Python 3.8+
- Webcam

## ⚡ Instalação Rápida

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd sleepy_detector
```

2. **Crie ambiente virtual**
```bash
python -m venv .venv
```

3. **Ative o ambiente virtual**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. **Instale dependências**
```bash
pip install -r requirements.txt
```

5. **Execute o sistema**
```bash
python launcher_gui.py
```

## 🎯 Como Usar

### Uso Básico (Detecção Imediata)
1. Execute `python launcher_gui.py`
2. Clique em **"👁️ DETECTOR DE OLHOS (PRINCIPAL)"**
3. Posicione-se na frente da webcam

### Treinamento Personalizado
1. **Organize Dataset**: Use "📁 ORGANIZAR DATASET DE OLHOS"
2. **Treine Modelos**: Use "🤖 TREINAR MODELOS DE OLHOS"
3. **Execute Detector**: Use "👁️ DETECTOR DE OLHOS (PRINCIPAL)"

## 📁 Estrutura do Projeto

```
sleepy_detector/
├── launcher_gui.py              # 🚀 Interface principal
├── eye_fatigue_detector.py      # 👁️ Detector de fadiga
├── eye_dataset_organizer.py     # 📁 Organizador de dataset
├── eye_dataset_trainer.py       # 🤖 Treinador de modelos
├── haarcascade_*.xml           # 🔍 Detectores Haar Cascade
├── eye_calibration.json        # ⚙️ Configurações
├── requirements.txt            # 📦 Dependências
├── models/                     # 🤖 Modelos treinados
└── eye_dataset/               # 📊 Dataset de olhos
    ├── alert/                 # Olhos alerta
    └── drowsy/               # Olhos sonolência
```

## 🛠️ Funcionalidades

- **Detecção em Tempo Real**: Análise contínua via webcam
- **Extração de Olhos**: Foco apenas nas regiões oculares
- **Machine Learning**: Modelos RandomForest e SVM
- **Calibração Automática**: Ajuste para diferentes usuários
- **Interface Amigável**: GUI com todas as funcionalidades

## 📊 Status do Sistema

Use "🔍 VERIFICAR STATUS DOS MODELOS" na interface para verificar:
- Status dos modelos treinados
- Arquivos essenciais
- Dataset disponível
- Configurações

## 🔧 Solução de Problemas

### Erro de Dependências
```bash
pip install -r requirements.txt
```

### Erro de Webcam
- Verifique se a webcam está conectada
- Feche outros aplicativos que usam a webcam

### Baixa Precisão
1. Use "🤖 TREINAR MODELOS" com dataset personalizado
2. Certifique-se de ter boa iluminação
3. Posicione-se adequadamente na frente da câmera

## 🎯 Tecnologias

- **OpenCV**: Processamento de imagem
- **Scikit-learn**: Machine Learning
- **NumPy**: Computação numérica
- **Tkinter**: Interface gráfica
- **Matplotlib**: Visualização de dados

## 📝 Licença

Este projeto está sob licença MIT.
