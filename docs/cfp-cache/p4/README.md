# CFP cache — resultado da etapa P4

Data: 2026-09-09. **P4 concluída; P5–P6 pendentes.**

A preparação em SSD agora pode usar processos independentes de CPU, com entradas pendentes limitadas, uma cota compartilhada de disco e um único coordenador do manifesto. A execução sequencial continua sendo o padrão. **559 testes passaram**, incluindo **36 novos testes da P4**, sem falhas, erros ou skips.

Nas três imagens originais de 2748 × 2748, dois trabalhadores reduziram o tempo de preparação em **27.5%** frente a um trabalhador. Todos os tensores brutos coincidiram com a P3 e as estatísticas com a P1. Os formatos de cache e checkpoints, as APIs legadas e a matemática da CFP foram preservados.

## Implementação e uso

O [coordenador paralelo](../../../mtlearn/python/mtlearn/layers/cfp/preparation/_parallel_preparation.py) recebe imagens canônicas uint8 do processo principal. As fontes, transformações, máscaras, parâmetros e normalizadores não são serializados. Isso permite usar datasets locais definidos no notebook. Cada tarefa corresponde a uma imagem com seus canais/árvores; apenas entradas ausentes são construídas.

```python
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore

preprocessor = CFPPreprocessor.from_layer(linear_layer)
with DiskStore("cfp-cache", max_disk_bytes=4 * 1024**3) as store:
    result = preprocessor.prepare(
        train_dataset,
        store=store,
        manifest="train-v1",
        source_version="dataset-split-v1",
        preprocessing_version="original-resolution-u8-v1",
        collect_stats=linear_layer.get_statistics_contract(),
        num_workers=2,
        max_in_flight=2,
        worker_threads=1,
        max_sample_bytes=8 * 1024**2,
        max_retries=0,
    )
    linear_layer.set_stats(result.statistics)
    mlp_layer.set_stats(result.statistics)
```

As cotas do exemplo são ilustrativas. Em scripts, criar a aplicação e chamar a preparação dentro de `main()` protegido por `if __name__ == "__main__":`. O pacote deve estar disponível no mesmo ambiente dos processos. O [guia público](../../source/guides/connected-filter-preprocessing.md#bounded-cpu-preparation-processes) documenta todos os parâmetros e limites.

- `num_workers=0` mantém a execução sequencial para qualquer store. A opção positiva desta entrega exige `DiskStore`; RAM e ausência de retenção continuam disponíveis sequencialmente.
- `max_in_flight` limita imagens ativas mais pendentes e os inputs guardados para retry. O padrão é o número de trabalhadores. `max_sample_bytes` limita cada imagem canônica, incluindo canais; seu padrão é 64 MiB.
- O processo principal lê/quantiza a fonte. Os trabalhadores recebem somente arrays CPU e descrições primitivas, sem camada, estatísticas ou acelerador. Descritores de arquivos com leitura no trabalhador ficam para evolução posterior.
- Cada trabalhador mantém uma entrada bruta por vez, grava um temporário exclusivo e aguarda a publicação. As gravações usam contador compartilhado sob trava; a construção de árvores e atributos permanece paralela. A cota inclui arquivos válidos, inválidos e temporários, sem remover entradas válidas automaticamente.
- O coordenador verifica tamanho, checksums, identidade, topologia e atributos antes de publicar atomicamente e registrar no SQLite. Os trabalhadores nunca abrem uma conexão SQLite.
- Conteúdos idênticos em voo aguardam sua preparação existente. Contribuições são únicas por amostra/canal/árvore. A redução estatística percorre a ordem fixa do manifesto após a preparação, antes do treinamento; não acrescenta trabalho às épocas.

A conversão uint8, topologia, resíduos, atributos, ordens de percurso, momentos e normalização permanecem os da referência. As identidades e a versão do formato não mudaram: os tensores do SSD da P3 são reutilizáveis na P4. `PreparedDataset` continua com `DataLoader(num_workers=0)` para consumo; não criar pools de preparação dentro de workers de carregamento.

## Falhas e limites de memória

Cancelamento é consultado entre leituras e durante a espera por trabalhadores. Cancelar ou receber uma falha encerra os processos desta chamada e limpa seus temporários, preservando as entradas publicadas. O resultado cancelado não contém estatísticas parciais. Uma amostra pode terminar fora de ordem; repetir os mesmos contratos e IDs retoma o manifesto e verifica novamente a fonte e os arquivos.

`max_retries` repete erros ordinários sobre os mesmos pixels já quantizados, reaproveitando entradas publicadas. Cota excedida, falha de inicialização ou morte abrupta de trabalhador exigem retomada explícita. Uma proteção por travas de trabalhadores impede uma nova abertura para escrita enquanto processos de um coordenador encerrado abruptamente ainda possam gravar. Seus temporários são recuperados somente após esses processos saírem. Erros antes/depois de publicação continuam compatíveis com a recuperação da P3.

No contador compartilhado, a substituição de um arquivo inválido pode manter uma estimativa conservadora de uso até a próxima chamada. Retomar recalcula os bytes reais. SQLite/WAL, blocos do filesystem e caches do sistema operacional não fazem parte da cota de arquivos de tensores.

O limite de inputs no coordenador é `max_in_flight × max_sample_bytes`. Acrescentar cópias de IPC/trabalhadores, um item transitório da fonte e quantização, memória nativa de árvores/atributos por trabalhador, validação de uma entrada no coordenador, retenção RAM opcional e page cache. **Esse limite não é um teto de RSS.** Em memória unificada, reservar também o consumo de treino/MPS.

A configuração usa `spawn` local, sem chamar `set_start_method` nem alterar threads ou variáveis de ambiente do processo principal. PyTorch recebe limites por trabalhador; inter-op fica em 1. OpenCV usa `setNumThreads(0)` para desativar paralelismo interno: no backend GCD desta máquina, limites positivos são ignorados e informam 10 threads. Testes confirmaram o limite efetivo de 1 no trabalhador. O backend morfológico atual não possui regiões OpenMP independentes do runtime do PyTorch.

## Validação

[Resultados estruturados](validation.json) e [XML da suíte](validation-tests.xml): **559/559 aprovados**, incluindo os 19 gradchecks existentes. A P4 acrescentou:

- 12 comparações com as fixtures P0, cobrindo árvores max/min/ToS, scorers linear/MLP e modos de normalização; estatísticas, saídas, loss e gradientes equivalentes;
- múltiplos canais/árvores em precisão dupla, comparando todos os tensores e resumos com execução sequencial;
- fonte local não serializável, máscara não serializável, fila de 3 para 2 processos, 17 amostras repetindo 3 conteúdos sem novas preparações duplicadas;
- ordem de conclusão forçada diferente da ordem do manifesto, com estatísticas exatamente iguais à redução sequencial;
- cancelamento com tarefas ativas, retomada sem criar processos quando todos os conteúdos existem e limites de configuração/entrada;
- cota compartilhada entre trabalhadores, sem ultrapassagem ou perda de entradas válidas;
- falha transitória real com retry, morte abrupta de trabalhador, corrupção entre serialização/publicação e morte abrupta do coordenador com trabalhador ainda ativo;
- inicialização CPU, limites efetivos de threads e ausência de parâmetros/normalizadores na configuração transportada.

Comando completo executado, a partir da raiz do repositório:

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p4.py test -- -q mtlearn/tests/python/test_bindings.py mtlearn/tests/python/test_data.py mtlearn/tests/python/test_cfp_preparation.py mtlearn/tests/python/test_cfp_prepared_runtime.py mtlearn/tests/python/test_cfp_disk_cache.py mtlearn/tests/python/test_cfp_parallel_preparation.py mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_deterministic.py mtlearn/tests/python/test_cfp_validation.py mtlearn/tests/python/test_cfp_cache.py mtlearn/tests/python/test_morphology_api.py mtlearn/tests/python/test_gradchecks.py mtlearn/tests/python/test_version.py mtlearn/tests/python/test_cfp_cache_baseline.py --junitxml=docs/cfp-cache/p4/validation-tests.xml
```

O driver de diagnóstico remove apenas em seus próprios processos o redirecionamento de uma instalação editável local para outro checkout. A biblioteca não contém essa adaptação. Imports do coordenador e de todos os trabalhadores, hash do backend, threads e estado de memória estão registrados em cada ensaio.

Também executados com sucesso:

```bash
/opt/anaconda3/bin/python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python/test_cfp_parallel_preparation.py scripts/benchmarks/cfp_cache_p4.py
git diff --check
```

As fixtures binárias da P0 tiveram todos os hashes conferidos. Os hashes dos arquivos da biblioteca registrados nos ensaios coincidem com os entregues; a extensão nativa permaneceu `c84f7e6cb8b9a6677324de94fd9a8372eedb16e3980ed931379b2caaf5a2ff6c`.

## Medição em resolução original

Máquina de referência: Apple M4, 10 CPUs lógicas, 16 GiB de memória unificada. Inputs `enhancement/0.png`, `enhancement/1000.png` e `enhancement/2040.png`, em tamanho original, com todos os 63 atributos e max-tree. A composição diagnóstica é a mesma da P1/P3; não representa o split integral do notebook. Preparação somente, sem época de treino ou leitura de máscaras neste ensaio.

| Processos CPU | Preparação fria | Retomada sem reconstrução | Pico RSS agregado | Memória disponível mínima |
| --- | ---: | ---: | ---: | ---: |
| 1 | 59.54 s | 0.78 s | 1.695 GiB | 3.599 GiB |
| 2 | 43.18 s | 0.79 s | 2.359 GiB | 3.361 GiB |

Ganho observado: **1.379×**, com uma execução por configuração. O tempo frio inclui leitura, criação dos processos, construção, atributos, resumos, serialização, validação/publicação e redução estatística. A diferença não demonstra escala linear nem prevê o tempo do dataset inteiro. O ensaio com dois trabalhadores mantém somente dois inputs em voo e uma thread PyTorch por trabalhador.

Cada store terminou com **3 entradas, 3 contribuições, 752,419,571 bytes de tensores (717.56 MiB)**, abaixo da cota de 1 GiB. Retenção RAM final: **0 bytes**. As retomadas não abriram processos nem reconstruíram/resumiram entradas. Checksums de **todos os tensores brutos** coincidiram com a P3, e todos os momentos coincidiram com a P1 dentro de `rtol=1e-10`, `atol=1e-12`. O armazenamento continua independente do scorer.

O monitor amostrou a soma do RSS do coordenador e de todos os descendentes a cada 100 ms, incluindo trabalhadores, resource tracker e os comandos curtos de proveniência. O supervisor externo fica fora dessa soma. Páginas compartilhadas podem ser contadas mais de uma vez; picos mais curtos que o intervalo podem não aparecer. RSS e memória disponível física são relatados separadamente. Limites de interrupção: 4 GiB de RSS agregado, 1 GiB mínimo disponível e 600 s. Nenhum foi atingido.

Quatro trabalhadores não foram executados: quatro vezes o maior pico individual observado de trabalhador já projeta **5.13 GiB**, acima do teto de 4 GiB antes de incluir o coordenador. Esse cálculo é conservador e não uma medição. Usar inicialmente 1 trabalhador e passar para 2 após verificar a margem local; não escolher o número de processos apenas pelos núcleos disponíveis.

Comandos dos ensaios:

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p4.py --num-workers 1 --output docs/cfp-cache/p4/runs/one_worker --cache-dir build/cfp-cache-p4-one
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p4.py --num-workers 2 --output docs/cfp-cache/p4/runs/two_workers --cache-dir build/cfp-cache-p4-two
```

Para repetir uma medição fria, escolher novos diretórios de saída e de cache. O comparativo usa `build/cfp-cache-p3-validated` como referência local dos tensores. [Resumo](summary.json), [CSV](measurements.csv), [ensaio com 1 processo](runs/one_worker/) e [ensaio com 2 processos](runs/two_workers/) contêm os resultados e a proveniência.

Não executados nesta etapa: dataset integral, quatro processos, Linux/Windows, CUDA indisponível e época em resolução original. Os testes existentes de MPS passaram; os trabalhadores novos executam somente CPU. A **P5** continua pendente: adaptar o notebook ao armazenamento compartilhado e às métricas por lotes. A **P6** fecha a documentação geral e a estabilização.
