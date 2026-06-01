Aqui está um modelo de `README.md` simples, conciso, de caráter científico e profissional para o seu repositório no GitHub.

---

# Pipeline de Curadoria SOTA e Treinamento de DCGAN no Dataset CelebA

Este repositório contém uma abordagem em duas etapas para filtragem inteligente de dados (curadoria de dados) e síntese de imagens utilizando Redes Adversariais Convolutivas Profundas (DCGAN) no conjunto de dados CelebA. O objetivo é mitigar o impacto de *outliers*, inconsistências de pose e ruído semântico na convergência e qualidade do modelo gerador.

## Estrutura do Projeto

O projeto é dividido em dois módulos principais estruturados em Jupyter Notebooks:

1. **`curadoria2.ipynb`**: Implementação do pipeline de filtragem de dados baseado em modelos de estado da arte (SOTA).
2. **`notebook234290581f.ipynb`**: Arquitetura, treinamento e avaliação da DCGAN com cálculo nativo de FID.

---

## Módulo 1: Curadoria de Dados (`curadoria2.ipynb`)

Para otimizar o treinamento da GAN, o conjunto de dados original CelebA passa por uma filtragem rigorosa baseada em três critérios:

* **Similaridade Semântica (CLIP):** Extração de *embeddings* visuais para identificação e remoção de amostras com desvios semânticos do domínio principal.
* **Consistência de Pose:** Eliminação de variações extremas de ângulo e perfil facial por meio do cálculo de distância euclidiana ao centroide do espaço de características de pose.
* **Remoção de Outliers:** Aplicação de thresholds estatísticos sobre as distribuições de similaridade e pose (reduzindo o conjunto de dados inicial para aproximadamente 89.693 imagens curadas).

**Saída:** Geração de um arquivo estruturado (`filtered_celeba.txt`) contendo os caminhos absolutos das imagens validadas.

---

## Módulo 2: Arquitetura e Treinamento da DCGAN (`notebook234290581f.ipynb`)

O modelo generativo adota as diretrizes estritas do artigo original da arquitetura DCGAN implementada em PyTorch.

### Hiperparâmetros de Treinamento

* **Dimensão Latente ($z$):** 100
* **Resolução de Saída:** 64 $\times$ 64 pixels (RGB)
* **Tamanho do Batch:** 128
* **Taxa de Aprendizado (LR):** 0.0002
* **Otimizador:** Adam ($\beta_1 = 0.5, \beta_2 = 0.999$)
* **Épocas:** 100

### Monitoramento e Avaliação

O progresso do gerador é quantificado por meio do cálculo periódico do **Fréchet Inception Distance (FID)** integrado nativamente ao fluxo de execução. O cálculo adota uma subamostra padrão de 5.000 imagens reais do conjunto filtrado para comparação de distribuições no espaço de features da Inception-v3.

---

## Pré-requisitos e Dependências

O ambiente foi otimizado para execução em aceleradores de hardware (NVIDIA Tesla T4). As principais dependências incluem:

* Python 3.12+
* PyTorch / Torchvision
* OpenAI CLIP
* NumPy / Pandas
* Matplotlib / PIL

Para instalar a extensão do CLIP necessária para o pipeline de curadoria:

```bash
pip install git+https://github.com/openai/CLIP.git

```

*(Nota: As dependências restantes seguem os pacotes padrão do ambiente Kaggle Production Docker)*.

---

## Execução

1. Execute o notebook `curadoria2.ipynb` para processar o dataset CelebA e gerar a lista indexada `filtered_celeba.txt`.
2. Configure o caminho do arquivo gerado na variável `FILTERED_LIST` do arquivo `notebook234290581f.ipynb`.
3. Execute o notebook `notebook234290581f.ipynb` para iniciar o treinamento da DCGAN e gerar os logs de perda e métricas de FID.
