# Sistema de Detecção de Fadiga

Sistema para detectar sonolência através da análise dos olhos usando Machine Learning.

## Características

- **Detecção através dos olhos**: Analisa as regiões dos olhos para identificar sono
- **Monitoramento da cabeça**: Detecta quando a cabeça está baixa por muito tempo
- **Alertas temporais**: Avisa após 3 segundos com olhos fechados ou 5 segundos com cabeça baixa
- **Alertas sonoros**: Emite sons de alerta quando detecta sonolência
- **Interface gráfica**: Interface simples para usar todas as funcionalidades
- **Dataset personalizado**: Permite treinar com suas próprias imagens
- **Modelos de ML**: Usa RandomForest e SVM para classificação
- **Calibração automática**: Se ajusta à posição normal da sua cabeça

## Recursos de Alerta

### Detecção de Olhos Fechados
- **Tempo**: Emite alerta após 3 segundos com olhos fechados
- **Como funciona**: Combina detecção de olhos com análise de probabilidade
- **Resposta**: Mostra alerta na tela, emite som e exibe mensagem no terminal

### Detecção de Cabeça Baixa  
- **Tempo**: Emite alerta após 5 segundos com cabeça baixa
- **Como funciona**: Analisa a posição da face na imagem
- **Calibração**: Se calibra automaticamente nos primeiros segundos
- **Resposta**: Mostra alerta na tela, emite som e exibe mensagem no terminal

## Requisitos

- Python 3.8 ou superior
- Webcam

## Instalação

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd fatigue-detector
```

2. **Crie um ambiente virtual**
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

4. **Instale as dependências**
```bash
pip install -r requirements.txt
```

5. **Execute o programa**
```bash
python launcher_gui.py
```

## Como Usar

### Para usar imediatamente
1. Execute `python launcher_gui.py`
2. Clique em **"DETECTOR DE OLHOS (PRINCIPAL)"**
3. Posicione-se na frente da webcam

### Para treinar com seus próprios dados
1. **Organize o dataset**: Use "ORGANIZAR DATASET DE OLHOS"
2. **Treine os modelos**: Use "TREINAR MODELOS DE OLHOS"
3. **Execute o detector**: Use "DETECTOR DE OLHOS (PRINCIPAL)"

## Estrutura do Projeto

```
fatigue-detector/
├── launcher_gui.py              # Interface principal
├── eye_fatigue_detector.py      # Detector de fadiga
├── eye_dataset_organizer.py     # Organizador de dataset
├── eye_dataset_trainer.py       # Treinador de modelos
├── haarcascade_*.xml           # Detectores Haar Cascade
├── eye_calibration.json        # Configurações
├── requirements.txt            # Dependências
├── models/                     # Modelos treinados
└── eye_dataset/               # Dataset de olhos
    ├── alert/                 # Imagens de olhos alerta
    └── drowsy/               # Imagens de olhos sonolentos
```

## Funcionalidades

- **Detecção em tempo real**: Analisa continuamente através da webcam
- **Extração de regiões dos olhos**: Foca apenas nas áreas dos olhos
- **Machine Learning**: Usa modelos RandomForest e SVM para classificação
- **Calibração automática**: Se ajusta para diferentes usuários
- **Interface simples**: Interface gráfica com todas as funcionalidades

## Status do Sistema

Use "VERIFICAR STATUS DOS MODELOS" na interface para verificar:
- Status dos modelos treinados
- Arquivos necessários
- Dataset disponível
- Configurações atuais

## Controles Durante a Detecção

Quando o detector estiver rodando, você pode usar essas teclas:

- **`q`** - Sair do detector
- **`c`** - Calibrar o limite de detecção de sono
- **`s`** - Mostrar estatísticas em tempo real
- **`r`** - Resetar a posição de referência da cabeça

## Configurações de Alerta

### Tempos de alerta (podem ser modificados no código)
- **Olhos fechados**: 3.0 segundos
- **Cabeça baixa**: 5.0 segundos
- **Intervalo entre alertas**: 2.0 segundos

### Limite de posição da cabeça
- **Padrão**: 15% de variação na posição
- **Calibração**: Automática nos primeiros segundos

## Solução de Problemas

### Erro nas dependências
```bash
pip install -r requirements.txt
```

### Problema com a webcam
- Verifique se a webcam está conectada
- Feche outros programas que podem estar usando a webcam

### Alertas sonoros não funcionam
- Os alertas sonoros usam a biblioteca `winsound` (apenas Windows)
- Em outros sistemas operacionais, apenas os alertas visuais funcionam

### Problema com a posição da cabeça
- Mantenha a cabeça em posição normal nos primeiros segundos
- Use a tecla `r` para recalibrar se necessário

### Baixa precisão na detecção
1. Use "TREINAR MODELOS" com um dataset personalizado
2. Certifique-se de ter boa iluminação
3. Posicione-se adequadamente na frente da câmera

## Tecnologias Utilizadas

- **OpenCV**: Para processamento de imagem e vídeo
- **Scikit-learn**: Para os algoritmos de Machine Learning
- **NumPy**: Para computação numérica
- **Tkinter**: Para a interface gráfica
- **Matplotlib**: Para visualização de dados (opcional)

## Licença

Este projeto está sob licença MIT.
