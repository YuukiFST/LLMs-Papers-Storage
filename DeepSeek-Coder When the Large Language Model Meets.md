Reference: https://arxiv.org/pdf/2401.14196

## Documentação Técnica: DeepSeek-Coder e o Avanço da Inteligência de Código

### 1. Título Impactante
**DeepSeek-Coder: Rompendo Barreiras com Modelos de Código Open-Source de Nível SOTA**

### 2. Executive Summary
O relatório técnico "DeepSeek-Coder: When the Large Language Model Meets Programming" apresenta uma série de modelos de linguagem especializados em código (LLMs), variando de 1.3B a 33B parâmetros, desenvolvidos pela DeepSeek-AI. Este trabalho visa fechar a lacuna de desempenho entre modelos de código *open-source* e proprietários (como GPT-3.5/4 e Codex). O DeepSeek-Coder destaca-se pelo treinamento em 2 trilhões de tokens, incluindo dados em nível de repositório que preservam dependências entre arquivos, e pelo uso de uma estratégia de preenchimento no meio (Fill-in-the-Middle - FIM) otimizada. As avaliações demonstram que o modelo supera concorrentes *open-source* existentes (como CodeLlama e StarCoder) e atinge, ou até ultrapassa, o GPT-3.5 em benchmarks de geração de código, especialmente ao utilizar técnicas de *instruction tuning*.

### 3. Análise Técnica

#### 3.1. Metodologia e Construção de Dados
A chave para o desempenho do DeepSeek-Coder reside na qualidade e estrutura dos dados de treinamento, que compõem um total de 2 trilhões de tokens divididos em:
- **87%:** Código-fonte.
- **10%:** Texto em inglês relacionado a código (GitHub Markdown, StackExchange).
- **3%:** Texto em chinês não relacionado a código.

**Inovação em Nível de Repositório:**
Diferente de abordagens anteriores que treinavam em arquivos isolados, o DeepSeek-Coder implementa uma análise de dependência (Seção 2.2). Utilizando um algoritmo de *Topological Sort* (descrito no Algoritmo 1), os arquivos são reordenados para garantir que dependências (imports, includes) apareçam antes dos arquivos que as utilizam. Isso simula o fluxo real de trabalho em projetos de software, permitindo que o modelo gere código com melhor compreensão de contexto cruzado (cross-file context).

#### 3.2. Estratégia de Treinamento (FIM e Contexto)
O modelo utiliza dois objetivos de treinamento principais:
1.  **Next Token Prediction:** Padrão de previsão do próximo token.
2.  **Fill-in-the-Middle (FIM):** Essencial para tarefas de preenchimento de código no meio de um arquivo.

Experimentos de *ablation* (Seção 3.1.2) revelaram que configurar o FIM em 50% com o modo PSM (*Prefix-Suffix-Middle*) oferece o melhor equilíbrio. Taxas mais altas de FIM (100%) melhoraram a tarefa de preenchimento mas degradaram a geração de código padrão, indicando um trade-off crítico na política de treinamento.

**Janela de Contexto Estendida:**
Para lidar com projetos complexos, o contexto foi expandido para 16K tokens através do reescalonamento dos parâmetros RoPE (*Rotary Position Embedding*). Isso permite que o modelo processe arquivos maiores e dependências mais longas sem perda significativa de coerência.

#### 3.3. Arquitetura e Otimização
A série de modelos baseia-se na arquitetura DeepSeek-LLM, utilizando:
- **Decoder-only Transformer.**
- **FlashAttention v2** para eficiência computacional.
- **Grouped-Query-Attention (GQA)** no modelo de 33B para reduzir o custo de inferência.
- O otimizador AdamW foi utilizado com uma política de *learning rate* em três estágios.

#### 3.4. Avaliação de Resultados
O DeepSeek-Coder foi submetido a uma bateria extensa de testes:

- **HumanEval e MBPP:** O modelo *Base* 33B alcançou acurácia média de 50.3% e 66.0% respectivamente, superando o CodeLlama-34B em mais de 9%. O modelo *Instruct* 33B superou o GPT-3.5-Turbo no HumanEval.
- **DS-1000:** O modelo demonstrou proficiência no uso de bibliotecas de ciência de dados (NumPy, Pandas, PyTorch), validando sua capacidade em cenários práticos.
- **LeetCode Contest:** Em um benchmark curado para evitar contaminação de dados (criado entre julho/2023 e janeiro/2024), o DeepSeek-Coder-Instruct 33B superou todos os modelos *open-source* e o GPT-3.5-Turbo.
- **Raciocínio Matemático (Program-Based):** O modelo obteve pontuações competitivas em benchmarks como GSM8K e MATH, utilizando a abordagem PAL (*Program-Aided Math Reasoning*), demonstrando que código é um veículo eficaz para lógica matemática.

### 4. Key Takeaways
- **Dados Estruturados valem mais:** A ordenação de arquivos baseada em dependências (nível de repositório) melhora significativamente a capacidade de geração de código que envolve múltiplos arquivos.
- **Equilíbrio no FIM:** O uso excessivo da tarefa Fill-in-the-Middle (100%) pode prejudicar a geração de código padrão. Um equilíbrio de 50% no modo PSM é o ideal.
- **Eficiência Paramétrica:** O modelo de 6.7B performou tão bem quanto o CodeLlama-34B, sugerindo que a qualidade dos dados e a arquitetura podem compensar a diferença bruta de parâmetros.
- **Contexto Longo é Crucial:** A extensão para 16K tokens é viável e necessária para aplicações reais de engenharia de software.

### 5. Conclusão
O DeepSeek-Coder estabelece um novo estado da arte para modelos de código *open-source*. Ao focar na qualidade dos dados em nível de projeto e refinar estratégias de treinamento como o FIM, a DeepSeek demonstrou que é possível construir modelos que rivalizam com gigantes proprietários sem custos proibitivos de API. A disponibilidade sob licença permissiva incentiva a pesquisa e o desenvolvimento comercial de ferramentas avançadas de inteligência de código.

---

### ETAPA FINAL: PROMPT IMPROVEMENT MODE

A análise do documento (Seção 4.1, *LeetCode Contest Benchmark*) identificou uma heurística específica de engenharia de prompt que melhora o desempenho do modelo em tarefas complexas de codificação.

**HEURÍSTICA IDENTIFICADA:** *Chain-of-Thought (CoT) forçado antes da geração de código.*
O documento observa que pedir ao modelo para escrever um *outline* passo a passo antes do código ajuda na compreensão de dependências e lógica, especialmente em tarefas difíceis.

#### 1. PROMPT ORIGINAL
```text
{problem_description}

Please complete the code below to solve the above problem:
```python
{code_template}
```
```

#### 2. VERSÃO MELHORADA
```text
{problem_description}

You need first to write a step-by-step outline and then write the code.

Please complete the code below to solve the above problem:
```python
{code_template}
```
```

#### 3. JUSTIFICATIVAS TÉCNICAS
- **Incremento de Performance:** Conforme os resultados na Tabela 5 do documento, o uso de CoT (+CoT) melhorou consistentemente as pontuações nos níveis de dificuldade "Medium" e "Hard" para os modelos DeepSeek-Coder-Instruct (ex: de 12.1% para 17.6% no nível Médio para o modelo 6.7B).
- **Decomposição de Problemas:** A instrução "write a step-by-step outline" força o modelo a decompor a lógica algébrica antes de traduzi-la para sintaxe de programação. Isso reduz a taxa de erro em dependências lógicas complexas.
- **Contexto de Raciocínio:** O processo de criar o esboço (outline) fornece um contexto de raciocínio interno (ou explícito no contexto) que guia a geração subsequente do código, mitigando alucinações lógicas.
