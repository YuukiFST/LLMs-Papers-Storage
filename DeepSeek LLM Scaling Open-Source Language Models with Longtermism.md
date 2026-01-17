Reference: https://arxiv.org/pdf/2401.02954

# DeepSeek LLM: Redefinindo as Leis de Escala e o Longtermismo em Modelos de Linguagem Open-Source

## Executive Summary

O documento **"DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"** apresenta a iniciativa da DeepSeek-AI para desenvolver modelos de linguagem de larga escala (7B e 67B parâmetros) com uma perspectiva de longo prazo. O estudo foca na reavaliação das **Leis de Escala (Scaling Laws)**, introduzindo novas métricas e descobertas sobre o equilíbrio ideal entre tamanho do modelo e volume de dados.

A pesquisa culmina na criação do DeepSeek LLM, treinado em 2 trilhões de tokens, que supera o LLaMA-2 70B em benchmarks de código, matemática e raciocínio. Através de uma arquitetura otimizada e estratégias de alinhamento (SFT e DPO), o modelo *chat* de 67B demonstrou desempenho superior ao GPT-3.5 em avaliações abertas, consolidando-se como uma contribuição robusta para o ecossistema open-source.

---

## Análise Técnica

### 1. Metodologia de Pré-treinamento e Dados

A base do desempenho do DeepSeek LLM reside na qualidade e escala dos dados. A equipe adotou uma abordagem rigorosa em três estágios: deduplicação, filtragem e remixagem.

*   **Estratégia de Deduplicação:** Foi implementada uma estratégia "agressiva", deduplicando não apenas dentro de um *dump* do Common Crawl, mas *entre* 91 *dumps* distintos. Isso resultou em uma taxa de remoção de 89.8%, significativamente superior à deduplicação isolada (Tabela 1).
*   **Tokenização:** Utilizou-se o algoritmo **Byte-level Byte-Pair Encoding (BBPE)**. O vocabulário foi treinado em um corpus multilingue de 24GB, totalizando 102.400 tokens para incluir especiais e reservar espaço para expansão.
*   **Remixagem:** Ajuste na distribuição de dados para corrigir desequilíbrios de domínios, garantindo representatividade e diversidade.

### 2. Arquitetura e Hiperparâmetros

Seguindo a tendência estabelecida pelo LLaMA, a DeepSeek fez ajustes macroestruturais focados em eficiência e performance de inferência (Seção 2).

*   **Design Macro:**
    *   **DeepSeek 7B:** 30 camadas.
    *   **DeepSeek 67B:** 95 camadas (priorizando profundidade em vez de largura) e utilizando **Grouped-Query Attention (GQA)** para otimizar o custo de inferência.
*   **Otimizador e Agendamento:**
    *   Otimizador AdamW ($\beta_1=0.9, \beta_2=0.95$).
    *   Substituição do agendador de taxa de aprendizado *Cosine* tradicional por um **Multi-step Learning Rate Scheduler**.
    *   **Justificativa:** Embora o desempenho final seja similar, o *multi-step* permite a reutilização da primeira fase de treinamento para contínuo treinamento (*continual training*), essencial para perspectivas de longo prazo.
    *   **Decaimento:** 2000 passos de *warmup*, seguido por redução para 31.6% após 80% dos tokens e 10% após 90% dos tokens.

### 3. Avanços nas Leis de Escala (Scaling Laws)

O estudo oferece contribuições significativas para a compreensão das leis de escala, corrigindo imprecisões de trabalhos anteriores (Seção 3).

*   **Escalonamento de Hiperparâmetros:**
    *   Através de experimentos, a equipe derivou relações de potência para o Tamanho do Lote ($B_{opt}$) e Taxa de Aprendizado ($\eta_{opt}$) baseados no orçamento de computação ($C$).
    *   $\eta_{opt} = 0.3118 \cdot C^{-0.1250}$
    *   $B_{opt} = 0.2920 \cdot C^{0.3271}$
    *   Isso permite definir hiperparâmetros ótimos para diferentes orçamentos de computação empiricamente.

*   **Nova Métrica de Escala do Modelo:**
    *   Crítica aos modelos anteriores que usavam parâmetros ($N$) como proxy de escala, ignorando custos de atenção e vocabulário.
    *   Proposta: **Non-embedding FLOPs/token ($M$)**.
    *   Fórmula refinada para computação: $C = M \cdot D$ (onde $D$ é o número de tokens), o que oferece previsões de perda de generalização mais precisas para modelos de larga escala.

*   **Impacto da Qualidade dos Dados:**
    *   Descobriu-se que a qualidade dos dados altera os expoentes de escala ($a$ para modelo, $b$ para dados).
    *   **Descoberta Crítica:** Quanto maior a qualidade dos dados, maior o expoente $a$ (escala do modelo). Isso sugere que, com dados de alta qualidade, é mais eficiente investir o aumento de computação no tamanho do modelo do que em mais dados.

### 4. Alinhamento (SFT e DPO)

Para tornar o modelo útil e seguro, foi empregado um pipeline de alinhamento em dois estágios (Seção 4).

*   **Supervised Fine-Tuning (SFT):**
    *   1.5M de instâncias (Inglês e Chinês).
    *   Estratégia de dois estágios para o modelo 7B para evitar repetição excessiva (comum em dados de matemática com padrões semelhantes). O modelo 67B não precisou do segundo estágio.
*   **Direct Preference Optimization (DPO):**
    *   Utilizado para refinar o comportamento do chat.
    *   Resultado: O DPO fortaleceu significativamente as habilidades de geração em open-ended sem prejudicar o desempenho em benchmarks padrão.

### 5. Avaliação de Resultados

O DeepSeek LLM foi submetido a avaliações rigorosas contra baselines fortes, como LLaMA-2 e GPT-3.5 (Seção 5).

*   **Modelo Base:** O DeepSeek 67B supera o LLaMA-2 70B em tarefas de matemática (GSM8K, MATH), código (HumanEval, MBPP) e raciocínio (BBH), mesmo treinando em um corpus bilíngue que pode diluir a performance em inglês puro comparado a modelos focados em um idioma.
*   **Modelo Chat:**
    *   **AlignBench (Chinês):** O DeepSeek 67B Chat superou o GPT-3.5 e ficou atrás apenas das versões do GPT-4, demonstrando proficiência superior em raciocínio e linguagem chinesa.
    *   **Avaliação Aberta (Inglês):** O desempenho foi comparado favoravelmente ao GPT-3.5, indicando alta qualidade nas respostas geradas.

---

## Key Takeaways

1.  **Multi-step Scheduler > Cosine para Longo Prazo:** Para projetos de IA com perspectiva de *continual training*, agendadores de taxa de aprendizado em degraus são mais práticos que os cossenoidais, pois facilitam a retomada e reutilização de fases de treinamento sem perda de performance.
2.  **Precisão na Métrica de Escala:** O uso de "Non-embedding FLOPs/token" ($M$) em vez de contagem de parâmetros ($N$) para calcular o orçamento de computação ($C$) reduz erros estatísticos nas previsões de lei de escala, especialmente em modelos menores.
3.  **Dados de Alta Qualidade Mudam a Estratégia:** Não existe uma proporção fixa universal entre modelo e dados (ex: 50/50). A qualidade dos dados dita a alocação ideal: dados de alta qualidade permitem justificar modelos maiores (menos dados, mais "neurônios").
4.  **Deducação Agressiva Vale a Pena:** Deduplicar em larga escala (transversalmente entre múltiplos dumps de dados) remove quase 90% do conteúdo duplicado, aumentando drasticamente a eficiência do treinamento por token único.
5.  **DPO como Padrão para Chat:** O uso de Direct Preference Optimization provou ser eficaz para melhorar a qualidade conversacional (*open-ended generation*) sem o custo de degradação de performance em benchmarks de conhecimento.

---

## Conclusão

O documento DeepSeek LLM estabelece um novo padrão para a pesquisa open-source ao unir a engenharia de dados meticulosa com uma investigação teórica aprofundada das leis de escala. A revisão da alocação de computação entre modelo e dados, fundamentada na qualidade do corpus, oferece um guia valioso para futuros treinamentos de LLMs.

A demonstração empírica de que um modelo 67B open-source pode superar o LLaMA-2 70B e competir com o GPT-3.5 valida a abordagem de "longtermismo" da DeepSeek-AI. A liberação de tais descobertas, juntamente com os pesos do modelo, acelera a democratização da IA de alto desempenho, fornecendo à comunidade não apenas um modelo poderoso, mas também o *know-how* metodológico para escalar além dele.

