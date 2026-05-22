# DCGAN para Geração de Faces no Dataset CelebA

## Visão Geral
Implementação de uma Deep Convolutional Generative Adversarial Network (DCGAN) em PyTorch para geração de imagens sintéticas de rostos. O modelo foi treinado utilizando um subconjunto de 20.000 imagens do dataset CelebA, com imagens redimensionadas para 64x64 pixels. O projeto inclui monitoramento quantitativo via Fréchet Inception Distance (FID) nativo, visualização de métricas de treinamento e geração de amostras finais.

## Principais Características
- Arquitetura DCGAN original com estabilização via Batch Normalization e label smoothing
- Cálculo nativo de FID para avaliação quantitativa contínua do gerador
- DataLoader personalizado otimizado para estruturas de diretório planas
- Visualização automática de perdas (Discriminador/Gerador) e curva de FID por época
- Geração e exportação automática de amostras sintéticas ao final do treinamento
- Otimizado para execução em GPU (compatível com Tesla T4)

## Arquitetura
**Gerador**
- Entrada: Vetor latente (100 dimensões, 1x1)
- 4 camadas transpostas convolucionais (1024 → 512 → 256 → 128 → 3 canais)
- Ativação: ReLU
- Normalização: BatchNorm2d
- Saída: Imagem 64x64 (Normalizada entre -1 e 1)

**Discriminador**
- Entrada: Imagem 64x64
- 4 camadas convolucionais (3 → 64 → 128 → 256 → 512 → 1 canal)
- Ativação: LeakyReLU (negative_slope=0.2)
- Normalização: BatchNorm2d
- Saída: Probabilidade binária (Sigmoid)

**Otimização e Perda**
- Otimizador: Adam (lr=2e-4, betas=(0.5, 0.999))
- Função de perda: Binary Cross-Entropy com suavização de rótulos (real=0.9, fake=0.0)
- Épocas: 100 | Batch Size: 128 | Dimensão Latente: 100

## Dataset
- **Fonte:** `jessicali9530/celeba-dataset` (img_align_celeba)
- **Quantidade:** 20.000 imagens
- **Pré-processamento:** `CenterCrop(140) → Resize(64) → ToTensor → Normalize(mean=[0.5], std=[0.5])`
- **Carregamento:** Dataset customizado (`CelebADataset`) para contornar a estrutura plana do dataset original do Kaggle

## Requisitos
```text
python >= 3.8
torch >= 1.10.0
torchvision >= 0.11.0
matplotlib
numpy
```

## Como Executar
1. Clone o repositório e acesse o diretório do projeto.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o notebook no ambiente local ou no Kaggle:
   ```bash
   jupyter notebook celeba-gan20k.ipynb
   ```
4. Certifique-se de que o dataset está disponível em `/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba` ou ajuste a variável `BASE_DIR` conforme o caminho local.
5. Execute as células sequencialmente. As métricas e as imagens geradas serão salvas automaticamente na pasta `./output_dcgan`.

## Monitoramento e Resultados
Durante o treinamento, o notebook gera e salva automaticamente:
- `training_plot.png`: Curvas de perda do Discriminador e Gerador, junto com a evolução do FID por época
- `real_samples.png`: Batch de imagens reais do dataset para referência
- `final_generated.png`: 64 amostras geradas pelo gerador após 100 épocas

O treinamento convergiu para um FID final aproximado de ~18-19, indicando boa qualidade de geração e estabilidade adversarial. Os logs de perda e FID são exportados em formato PNG para análise posterior.

## Licença
Este projeto é disponibilizado para fins educacionais e de pesquisa. Consulte os termos de uso do dataset CelebA original para restrições de uso comercial.
