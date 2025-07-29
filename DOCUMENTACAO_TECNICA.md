# Documentação Técnica - Sistema de Detecção de Fadiga

## Visão Geral do Projeto

Este é um sistema completo de **detecção de fadiga em tempo real** que monitora os olhos e a posição da cabeça através da webcam, alertando quando detecta sinais de sonolência.

## Estrutura e Arquivos do Projeto

### **1. `launcher_gui.py` - Interface Principal**
- **Função**: Interface gráfica principal do sistema
- **Tecnologia**: Tkinter (GUI nativa do Python)
- **Responsabilidades**:
  - Verificar dependências instaladas (OpenCV, NumPy, etc.)
  - Lançar outros módulos do sistema
  - Verificar status dos modelos treinados
  - Fornecer instruções de uso

### **2. `eye_dataset_organizer.py` - Organizador de Dataset**
- **Função**: Prepara dados para treinamento
- **Processo**:
  1. Lê datasets com estruturas variadas (drowsy/alert, open_eyes/closed_eyes)
  2. Detecta faces usando Haarcascade
  3. Extrai regiões dos olhos de cada face
  4. Redimensiona para 64x32 pixels
  5. Organiza em pastas `alert/` e `drowsy/`

### **3. `eye_dataset_trainer.py` - Treinador de Modelos**
- **Função**: Treina modelos de machine learning
- **Algoritmos**: RandomForest e SVM
- **Características extraídas**:
  - **Estatísticas básicas**: média, desvio padrão, variância
  - **Histograma**: distribuição de intensidades
  - **Gradientes**: bordas e texturas (Sobel X e Y)
  - **LBP (Local Binary Patterns)**: padrões locais de textura
- **Saída**: Modelos salvos em `models/eye_fatigue_models.pkl`

### **4. `eye_fatigue_detector.py` - Detector Principal**
- **Função**: Detecção em tempo real
- **Características**:
  - Detecção de faces e olhos
  - Análise de probabilidade de sonolência
  - Monitoramento de posição da cabeça
  - Alertas temporais (3s olhos fechados, 5s cabeça baixa)
  - Alertas sonoros e visuais

## Como Funciona o Sistema Haarcascade

### **O que é Haarcascade?**
- **Definição**: Classificador em cascata baseado em características de Haar
- **Inventor**: Paul Viola e Michael Jones (2001)
- **Princípio**: Detecta objetos usando padrões de luz e sombra

### **Como Funciona:**

1. **Características de Haar**: 
   - Padrões retangulares simples (claro-escuro)
   - Exemplo: olhos têm região escura entre duas claras (sobrancelhas e bochechas)

2. **Classificador em Cascata**:
   - Múltiplos estágios de classificação
   - Cada estágio elimina rapidamente regiões não-face
   - Apenas regiões promissoras passam para próximo estágio

3. **Treinamento**:
   - Milhares de imagens positivas (faces) e negativas (não-faces)
   - Algoritmo AdaBoost seleciona melhores características
   - Processo intensivo (semanas de treinamento)

### **Arquivos Haarcascade no Projeto:**

```python
# Estes arquivos SÃO PRÉ-TREINADOS pelo OpenCV
"haarcascade_frontalface_default.xml"  # Detecta faces frontais
"haarcascade_eye.xml"                  # Detecta olhos
```

**⚠️ IMPORTANTE**: Estes arquivos **NÃO são gerados pelo projeto**. São arquivos pré-treinados que vêm com o OpenCV, resultado de anos de pesquisa e treinamento em milhões de imagens.

## Fluxo Completo do Sistema

### **Fase 1: Preparação dos Dados**
```
Dataset Raw → eye_dataset_organizer.py → Imagens de olhos organizadas
```

### **Fase 2: Treinamento**
```
Imagens organizadas → eye_dataset_trainer.py → Modelos ML treinados
```

### **Fase 3: Detecção em Tempo Real**
```
Webcam → Haarcascade (faces/olhos) → Modelos ML → Alertas
```

## Algoritmo de Detecção Detalhado

### **1. Captura e Pré-processamento**
```python
# Captura frame da webcam
frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

### **2. Detecção de Face**
```python
# Haarcascade detecta faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

### **3. Detecção de Olhos**
```python
# Para cada face, detecta olhos
eyes = eye_cascade.detectMultiScale(face_roi)
```

### **4. Análise de Fadiga**
```python
# Extrai características de cada olho
features = extract_features(eye_image)
# Prediz probabilidade de sonolência
sleep_prob = model.predict_proba(features)[0][1]
```

### **5. Sistema de Alertas Temporais**
```python
# Se probabilidade alta de sono
if sleep_prob > threshold:
    if olhos_fechados_inicio is None:
        olhos_fechados_inicio = time.time()
    elif time.time() - olhos_fechados_inicio > 3.0:
        # ALERTA: Olhos fechados há mais de 3 segundos
```

## Características Técnicas Avançadas

### **Extração de Características**
1. **Estatísticas**: Captura brilho geral do olho
2. **Histograma**: Distribuição de tons (olhos fechados têm padrão diferente)
3. **Gradientes**: Bordas das pálpebras e íris
4. **LBP**: Texturas locais (cílios, rugas, etc.)

### **Sistema de Calibração**
- **Automática**: Aprende posição normal da cabeça
- **Personalizada**: Ajusta threshold por usuário
- **Adaptativa**: Melhora com o uso

### **Prevenção de Falsos Positivos**
- **Histórico temporal**: Suaviza detecções
- **Múltiplos olhos**: Confirma detecção
- **Thresholds adaptativos**: Ajusta sensibilidade

## Configurações e Parâmetros

```python
# Tempos de alerta
tempo_alerta_olhos = 3.0      # 3 segundos
tempo_alerta_cabeca = 5.0     # 5 segundos

# Sensibilidade
threshold_padrao = 0.7        # 70% probabilidade
threshold_cabeca_baixa = 0.12 # 12% deslocamento

# Histórico para suavização
max_historico = 8             # 8 frames
```

## Dependências e Tecnologias

- **OpenCV**: Visão computacional e Haarcascades
- **Scikit-learn**: Algoritmos de ML (RandomForest, SVM)
- **NumPy**: Processamento numérico
- **Matplotlib**: Visualizações (opcional)
- **Tkinter**: Interface gráfica
- **Threading**: Alertas sonoros assíncronos

## Casos de Uso

1. **Motoristas**: Prevenir acidentes por sonolência
2. **Trabalhadores**: Monitorar fadiga em turnos longos
3. **Estudantes**: Detectar perda de atenção
4. **Operadores**: Segurança em trabalhos críticos

## Detalhes da Implementação

### **Extração de Características dos Olhos**

O sistema extrai múltiplas características de cada região de olho detectada:

```python
def extract_features(eye_image):
    features = []
    
    # 1. Estatísticas básicas
    features.extend([
        np.mean(eye_image),
        np.std(eye_image),
        np.var(eye_image)
    ])
    
    # 2. Histograma (distribuição de intensidades)
    hist = cv2.calcHist([eye_image], [0], None, [16], [0, 256])
    features.extend(hist.flatten())
    
    # 3. Gradientes (bordas)
    sobelx = cv2.Sobel(eye_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(eye_image, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([np.mean(np.abs(sobelx)), np.mean(np.abs(sobely))])
    
    # 4. LBP (padrões de textura local)
    lbp = local_binary_pattern(eye_image, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10)
    features.extend(lbp_hist)
    
    return np.array(features)
```

### **Sistema de Alerta Temporal**

O sistema mantém controle temporal para evitar falsos positivos:

```python
def gerenciar_alertas_temporais(self, sleep_prob, head_position):
    current_time = time.time()
    
    # Detecção de olhos fechados
    if sleep_prob > self.threshold_sono:
        if self.olhos_fechados_inicio is None:
            self.olhos_fechados_inicio = current_time
        elif current_time - self.olhos_fechados_inicio > 3.0:
            if self.ultimo_alerta_olhos is None or current_time - self.ultimo_alerta_olhos > 2.0:
                self.emitir_alerta_sonoro("olhos")
                self.ultimo_alerta_olhos = current_time
    else:
        self.olhos_fechados_inicio = None
    
    # Detecção de cabeça baixa
    if head_position > self.threshold_cabeca_baixa:
        if self.cabeca_baixa_inicio is None:
            self.cabeca_baixa_inicio = current_time
        elif current_time - self.cabeca_baixa_inicio > 5.0:
            if self.ultimo_alerta_cabeca is None or current_time - self.ultimo_alerta_cabeca > 2.0:
                self.emitir_alerta_sonoro("cabeca")
                self.ultimo_alerta_cabeca = current_time
    else:
        self.cabeca_baixa_inicio = None
```

### **Calibração Automática da Posição da Cabeça**

O sistema aprende automaticamente a posição normal da cabeça:

```python
def analisar_posicao_cabeca(self, face_y, frame_height):
    posicao_relativa = face_y / frame_height
    
    # Calibração automática nos primeiros frames
    if len(self.historico_posicao_cabeca) < 30:
        self.historico_posicao_cabeca.append(posicao_relativa)
        if len(self.historico_posicao_cabeca) == 30:
            self.posicao_referencia_cabeca = np.mean(self.historico_posicao_cabeca)
    
    # Calcula desvio da posição de referência
    if self.posicao_referencia_cabeca is not None:
        desvio = abs(posicao_relativa - self.posicao_referencia_cabeca)
        return desvio
    
    return 0.0
```

## Otimizações de Performance

### **Processamento de Múltiplos Olhos**
O sistema processa todos os olhos detectados e usa a média das probabilidades:

```python
sleep_probs = []
for (ex, ey, ew, eh) in eyes:
    eye_roi = face_gray[ey:ey+eh, ex:ex+ew]
    eye_resized = cv2.resize(eye_roi, (64, 32))
    features = self.extract_features(eye_resized)
    prob = self.model.predict_proba([features])[0][1]
    sleep_probs.append(prob)

# Usa a média das probabilidades
avg_sleep_prob = np.mean(sleep_probs) if sleep_probs else 0.0
```

### **Suavização Temporal**
Mantém histórico de detecções para suavizar resultados:

```python
# Adiciona ao histórico
self.historico_deteccoes.append(avg_sleep_prob)
if len(self.historico_deteccoes) > 8:
    self.historico_deteccoes.pop(0)

# Calcula probabilidade suavizada
sleep_prob_suavizada = np.mean(self.historico_deteccoes)
```

O sistema combina técnicas clássicas de visão computacional (Haarcascade) com machine learning moderno, criando uma solução robusta e em tempo real para detecção de fadiga!
