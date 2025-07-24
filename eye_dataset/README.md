# 👁️ INSTRUÇÕES PARA DATASET DE OLHOS

## 📁 Estrutura criada:

```
eye_dataset/
├── alert/    ← COLOQUE AQUI: Imagens de olhos ABERTOS/ALERTAS
└── drowsy/   ← COLOQUE AQUI: Imagens de olhos FECHADOS/SONOLENTOS
```

## 🎯 Como usar:

1. **Coloque suas imagens de olhos**:
   - Olhos abertos/alertas → pasta `alert/`
   - Olhos fechados/sonolentos → pasta `drowsy/`

2. **Formatos suportados**:
   - .jpg, .jpeg, .png, .bmp

3. **Execute o treinamento**:
   - Abra a interface gráfica
   - Clique em "🤖 TREINAR MODELOS DE OLHOS"
   - Informe o caminho: `eye_dataset`

## 📊 Recomendações:

- **Quantidade**: Pelo menos 100+ imagens por categoria
- **Qualidade**: Imagens claras dos olhos
- **Variedade**: Diferentes pessoas, ângulos, iluminação
- **Proporção**: Tente ter quantidades similares (alert ≈ drowsy)

## ✅ Depois do treinamento:

- Use o "👁️ DETECTOR DE OLHOS" para testar
- Espere acurácia > 90%!
