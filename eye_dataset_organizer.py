"""
Organizador de Dataset de Olhos
Ajuda a extrair e organizar olhos de datasets existentes
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import json
from datetime import datetime

class EyeDatasetOrganizer:
    """
    Organiza datasets focando na extração de regiões de olhos
    """
    
    def __init__(self):
        print("👁️ Organizador de Dataset de Olhos")
        print("🎯 Extrai e organiza regiões de olhos de datasets existentes")
        
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        
        self.estatisticas = {
            'imagens_processadas': 0,
            'faces_detectadas': 0,
            'olhos_extraidos': 0,
            'alertas_salvos': 0,
            'sono_salvos': 0
        }
    
    def detectar_estrutura_dataset(self, dataset_path):
        """Detecta a estrutura do dataset existente"""
        print(f"\n📁 Analisando estrutura: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print("❌ Caminho não encontrado!")
            return None
        
        # Possíveis estruturas
        estruturas = [
            # Para datasets de fadiga
            (['drowsy', 'non_drowsy'], 'non_drowsy', 'drowsy'),
            (['alert', 'drowsy'], 'alert', 'drowsy'),
            (['awake', 'sleepy'], 'awake', 'sleepy'),
            
            # Para datasets de olhos
            (['open_eyes', 'closed_eyes'], 'open_eyes', 'closed_eyes'),
            (['open', 'closed'], 'open', 'closed'),
            
            # Outras possibilidades
            (['normal', 'tired'], 'normal', 'tired'),
            (['0', '1'], '0', '1'),  # Numeradas
        ]
        
        for pastas, pasta_alerta, pasta_sono in estruturas:
            if all(os.path.exists(os.path.join(dataset_path, p)) for p in pastas):
                alerta_count = len([f for f in os.listdir(os.path.join(dataset_path, pasta_alerta)) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                sono_count = len([f for f in os.listdir(os.path.join(dataset_path, pasta_sono)) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                
                print(f"✅ Estrutura detectada:")
                print(f"   📂 {pasta_alerta}: {alerta_count} imagens")
                print(f"   📂 {pasta_sono}: {sono_count} imagens")
                
                return {
                    'pasta_alerta': pasta_alerta,
                    'pasta_sono': pasta_sono,
                    'alerta_count': alerta_count,
                    'sono_count': sono_count
                }
        
        print("❌ Estrutura não reconhecida!")
        print("📋 Estruturas suportadas:")
        for pastas, _, _ in estruturas:
            print(f"   • {' / '.join(pastas)}")
        
        return None
    
    def extrair_olhos_de_imagem(self, img_path, output_folder, categoria, img_id):
        """
        Extrai regiões de olhos de uma imagem
        """
        try:
            # Carrega imagem
            img = cv2.imread(img_path)
            if img is None:
                return 0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detecta faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return 0
            
            self.estatisticas['faces_detectadas'] += 1
            
            # Para cada face
            olhos_salvos = 0
            for face_idx, (x, y, w, h) in enumerate(faces):
                roi_gray = gray[y:y+h, x:x+w]
                
                # Detecta olhos na face
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                
                # Salva cada olho
                for eye_idx, (ex, ey, ew, eh) in enumerate(eyes):
                    # Extrai região do olho
                    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Redimensiona para tamanho padrão
                    eye_resized = cv2.resize(eye_roi, (64, 32))
                    
                    # Nome do arquivo
                    filename = f"{categoria}_{img_id:06d}_face{face_idx}_eye{eye_idx}.png"
                    output_path = os.path.join(output_folder, filename)
                    
                    # Salva olho
                    cv2.imwrite(output_path, eye_resized)
                    olhos_salvos += 1
                    
                    self.estatisticas['olhos_extraidos'] += 1
                    
                    if categoria == 'alert':
                        self.estatisticas['alertas_salvos'] += 1
                    else:
                        self.estatisticas['sono_salvos'] += 1
            
            return olhos_salvos
            
        except Exception as e:
            print(f"⚠️ Erro ao processar {img_path}: {e}")
            return 0
    
    def organizar_dataset_olhos(self, dataset_path, output_path):
        """
        Organiza dataset extraindo apenas regiões de olhos
        """
        print(f"\n🔄 Organizando dataset de olhos...")
        print(f"📂 Origem: {dataset_path}")
        print(f"📂 Destino: {output_path}")
        
        # Detecta estrutura
        estrutura = self.detectar_estrutura_dataset(dataset_path)
        if not estrutura:
            return False
        
        # Cria pastas de output
        os.makedirs(output_path, exist_ok=True)
        
        # Mapeia categorias
        categorias = {
            estrutura['pasta_alerta']: 'alert',
            estrutura['pasta_sono']: 'drowsy'
        }
        
        # Cria subpastas
        for categoria in ['alert', 'drowsy']:
            categoria_path = os.path.join(output_path, categoria)
            os.makedirs(categoria_path, exist_ok=True)
            print(f"📁 Criada pasta: {categoria}")
        
        # Processa cada categoria
        for pasta_origem, categoria_destino in categorias.items():
            print(f"\n🔄 Processando categoria: {pasta_origem} -> {categoria_destino}")
            
            origem_path = os.path.join(dataset_path, pasta_origem)
            destino_path = os.path.join(output_path, categoria_destino)
            
            # Lista imagens
            imagens = [f for f in os.listdir(origem_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"📊 Encontradas {len(imagens)} imagens")
            
            # Processa cada imagem
            for idx, filename in enumerate(imagens):
                img_path = os.path.join(origem_path, filename)
                
                olhos_extraidos = self.extrair_olhos_de_imagem(
                    img_path, destino_path, categoria_destino, idx
                )
                
                self.estatisticas['imagens_processadas'] += 1
                
                # Progress
                if (idx + 1) % 100 == 0:
                    print(f"   Processadas: {idx + 1}/{len(imagens)} "
                          f"(Olhos extraídos: {olhos_extraidos})")
        
        # Salva estatísticas
        self.salvar_estatisticas(output_path)
        
        print(f"\n✅ ORGANIZAÇÃO CONCLUÍDA!")
        return True
    
    def salvar_estatisticas(self, output_path):
        """Salva estatísticas da organização"""
        stats = {
            'estatisticas': self.estatisticas,
            'data_organizacao': datetime.now().isoformat(),
            'estrutura_final': {
                'alert': self.estatisticas['alertas_salvos'],
                'drowsy': self.estatisticas['sono_salvos']
            }
        }
        
        stats_path = os.path.join(output_path, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Estatísticas salvas: {stats_path}")
    
    def mostrar_estatisticas(self):
        """Mostra estatísticas detalhadas"""
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"📷 Imagens processadas: {self.estatisticas['imagens_processadas']}")
        print(f"👤 Faces detectadas: {self.estatisticas['faces_detectadas']}")
        print(f"👁️ Olhos extraídos: {self.estatisticas['olhos_extraidos']}")
        print(f"🟢 Olhos alerta: {self.estatisticas['alertas_salvos']}")
        print(f"🔴 Olhos sono: {self.estatisticas['sono_salvos']}")
        
        if self.estatisticas['imagens_processadas'] > 0:
            taxa_faces = self.estatisticas['faces_detectadas'] / self.estatisticas['imagens_processadas']
            print(f"📈 Taxa detecção faces: {taxa_faces:.2f}")
        
        if self.estatisticas['faces_detectadas'] > 0:
            olhos_por_face = self.estatisticas['olhos_extraidos'] / self.estatisticas['faces_detectadas']
            print(f"👁️ Olhos por face: {olhos_por_face:.2f}")
    
    def visualizar_amostras(self, dataset_path, num_amostras=10):
        """Visualiza amostras do dataset organizado"""
        print(f"\n🖼️ Visualizando amostras de: {dataset_path}")
        
        for categoria in ['alert', 'drowsy']:
            categoria_path = os.path.join(dataset_path, categoria)
            
            if not os.path.exists(categoria_path):
                continue
            
            imagens = [f for f in os.listdir(categoria_path) 
                      if f.lower().endswith('.png')][:num_amostras]
            
            if not imagens:
                continue
            
            print(f"\n📂 Categoria: {categoria}")
            
            # Carrega e mostra imagens
            for i, img_name in enumerate(imagens):
                img_path = os.path.join(categoria_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    cv2.imshow(f'{categoria} - Amostra {i+1}', img)
            
            print(f"Pressione qualquer tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    """Função principal"""
    print("👁️ ORGANIZADOR DE DATASET DE OLHOS")
    print("=" * 50)
    
    organizador = EyeDatasetOrganizer()
    
    while True:
        print("\n📋 OPÇÕES:")
        print("1. Organizar dataset (extrair olhos)")
        print("2. Visualizar amostras de dataset")
        print("3. Mostrar estatísticas")
        print("4. Sair")
        
        opcao = input("\nEscolha uma opção (1-4): ").strip()
        
        if opcao == '1':
            print("\n📁 ORGANIZAÇÃO DE DATASET")
            
            dataset_origem = input("Caminho do dataset original: ").strip()
            if not dataset_origem or not os.path.exists(dataset_origem):
                print("❌ Caminho inválido!")
                continue
            
            dataset_destino = input("Caminho para salvar olhos extraídos: ").strip()
            if not dataset_destino:
                dataset_destino = os.path.join(os.path.dirname(dataset_origem), "eye_dataset")
                print(f"📂 Usando destino padrão: {dataset_destino}")
            
            sucesso = organizador.organizar_dataset_olhos(dataset_origem, dataset_destino)
            
            if sucesso:
                organizador.mostrar_estatisticas()
                print(f"\n🎉 Dataset de olhos criado em: {dataset_destino}")
                print("✅ Pronto para usar no treinamento!")
            
        elif opcao == '2':
            dataset_path = input("Caminho do dataset de olhos: ").strip()
            if os.path.exists(dataset_path):
                organizador.visualizar_amostras(dataset_path)
            else:
                print("❌ Caminho não encontrado!")
        
        elif opcao == '3':
            organizador.mostrar_estatisticas()
        
        elif opcao == '4':
            print("👋 Saindo...")
            break
        
        else:
            print("❌ Opção inválida!")

if __name__ == "__main__":
    main()
