# CFP cache — resultado da etapa P1

Data: 2026-09-09. **P1 concluída; P2–P6 pendentes.**

A CFP agora permite ajustar as estatísticas do treino em uma passagem, sem preencher o cache de imagens. A preparação morfológica bruta foi separada do modelo, e as estatísticas podem ser transferidas entre scorers compatíveis por snapshots. As constantes da normalização congelada são preparadas e reutilizadas, preservando as fórmulas e as tolerâncias da P0.

**Validação final: 352 testes aprovados, zero falhas e zero skips.** O ensaio final com três imagens de 2.748 × 2.748 pixels levou **64.77 segundos**, com **1.11 GiB de pico RSS observado**, nenhuma amostra no cache e 1.512 bytes de tensores estatísticos brutos ao final. Esses 1.512 bytes não representam toda a memória do processo: excluem constantes derivadas, parâmetros, objetos Python e temporários do runtime.

## 1. O que foi entregue

| Componente | Comportamento disponível |
| --- | --- |
| `PreparedMorphology` | Uma imagem/canal/árvore em CPU, com topologia, mapa pixel–nó, resíduos, ordens de percurso e atributos brutos; validação de esquema/valores e contagem de storages únicos. |
| `CFPPreprocessor.from_layer` | Extrai apenas os contratos de árvores e atributos. Não retém a camada, seus parâmetros, normalizador, acelerador ou imagens preparadas. |
| `prepare_image` / `prepare_u8` | Preparam uma imagem 2D temporária; a conversão canônica para uint8 e os tipos dos tensores permanecem os existentes. |
| `cfp.fit_stats(loader)` | Ajuste incremental, substituição explícita do ajuste anterior e congelamento sem preencher cache. Uma passagem vazia ou com falha preserva o estado anterior. |
| `cfp.get_stats()` / `cfp.set_stats(snapshot)` | Exportação e instalação com cópias defensivas de escalares CPU e validação dos contratos de árvore, atributos, dtype e normalização. |
| `StatisticsSnapshot.combine` | Combina snapshots compatíveis na ordem recebida; preserva momentos ponderados pelo número de nós e a multiplicidade das amostras. |
| `AttributeNormalizer.summarize` / `merge` | Resumos escalares independentes e redução incremental, sem guardar atributos de todas as imagens. |
| Normalização congelada | Reutiliza momentos CPU e constantes por dispositivo/dtype. Alterações de estatísticas ou da política invalidam os derivados. |
| Restauração de estado | Checkpoints e arquivos de estatísticas preservam os formatos anteriores; os momentos float64 permanecem em CPU, inclusive quando os parâmetros usam MPS. |

O `TreePayloadProvider` tornou-se um adaptador da preparação CPU para o payload legado. `build_dataloader_cached` e `build_dataloader_cached_fixed_stats` mantêm assinaturas, retornos e comportamento de pré-carregamento. As chaves dos parâmetros e contratos de inferência não receberam caminhos de cache, limites ou configuração de armazenamento.

Os arquivos principais estão em [preparation](../../../mtlearn/python/mtlearn/layers/cfp/preparation/), [normalização](../../../mtlearn/python/mtlearn/layers/cfp/normalization/), [camada CFP](../../../mtlearn/python/mtlearn/layers/cfp/connected_filter_preprocessing_layer.py) e [checkpoint helpers](../../../mtlearn/python/mtlearn/layers/checkpoint.py). O [guia público](../../source/guides/connected-filter-preprocessing.md#normalization-and-caching) inclui os novos exemplos. A distribuição de fontes passa a incluir as fixtures usadas pelos testes de referência.

## 2. Uso do ajuste sem retenção

Com `linear_layer`, `mlp_layer` e `trainloader` já configurados:

```python
snapshot = linear_layer.fit_stats(trainloader)
mlp_layer.set_stats(snapshot)

for images, targets in trainloader:
    outputs = linear_layer(images)
    # Calcular a perda e realizar o backward normalmente.
```

As duas camadas devem ter contratos compatíveis de árvores, atributos, dtype e normalização; pesos, nomes dos scorers e arquiteturas de scoring podem diferir. As estatísticas são copiadas; os parâmetros continuam independentes.

`fit_stats` usa **exatamente as amostras emitidas pelo carregador**, incluindo sampler, ordem, multiplicidade e `drop_last`. Ele não substitui silenciosamente o sampler nem inclui amostras omitidas pelo carregador. O chamador deve fornecer uma passagem finita sobre o treino e controlar seus workers/prefetch. O ajuste retém apenas o lote ativo e uma preparação por vez, além dos acumuladores escalares. Cada preparação e as referências ao lote são liberadas antes de solicitar os próximos dados.

No modo `none`, o snapshot tem estatísticas vazias e a semântica de atributos brutos permanece explícita. Nos modos estatísticos, as definições continuam sendo mínimos/máximos ou `count`, `sum` e `sumsq`, com momentos float64 em CPU. Não houve substituição por média das médias, mudança da variância, epsilon, clipping ou piso.

Combinação de partições do treino:

```python
from mtlearn.layers.cfp import StatisticsSnapshot

# Cada fit_stats representa um novo ajuste independente.
left = linear_layer.fit_stats(left_trainloader)
right = linear_layer.fit_stats(right_trainloader)
combined = StatisticsSnapshot.combine([left, right])
linear_layer.set_stats(combined)
```

A combinação não deduplica tentativas repetidas: duas amostras lógicas contam duas vezes. O controle de unicidade durante retomadas continua pertencendo à futura etapa do manifesto em SSD. `sample_count` informa imagens lógicas quando conhecido; checkpoints legados não guardam esse metadado e continuam sendo aceitos.

## 3. Contratos de preparação e normalização

A preparação usa `TreeSpec` e `FeatureSpec`, sem referências aos scorers. O objeto contém somente dados brutos em CPU, com índices int64, resíduos float32 e atributos no dtype configurado. Os mapas são protegidos contra alteração estrutural; os buffers dos tensores são compartilhados e somente leitura por contrato. Não se deve modificá-los in-place.

`validate()` verifica esquema, dimensões, tipos, dispositivo e ausência de gradientes. `validate(full=True)` também examina valores finitos, referências de nós, raiz e coerência das ordens/intervalos de percurso. `nbytes` conta storages únicos, incluindo memória por trás de views. O identificador opcional `input_id` é uma identidade lógica; não é um fingerprint que autorize reutilizar dados de pixels alterados. Fingerprints persistentes e integridade em disco permanecem na P3.

```python
from mtlearn.layers.cfp import CFPPreprocessor

preprocessor = CFPPreprocessor.from_layer(linear_layer)
prepared = preprocessor.prepare_image(image_2d)
prepared.validate(full=True)
print(prepared.image_shape, prepared.num_nodes, prepared.nbytes)
```

A normalização de estatísticas congeladas conserva os casts e a ordem matemática anteriores. O normalizador guarda somente os momentos e escalares derivados necessários, sem atributos normalizados por imagem no novo ajuste. `set_stats`, restauração, mudança de versão estatística e atualização explícita invalidam constantes; mudanças de política descartam os derivados da política anterior.

Os caches legados existentes não são apagados por `fit_stats` ou `set_stats`. Suas normalizações ficam inválidas e são atualizadas quando consumidas. `refresh_cached_normalization()` continua disponível para atualização explícita. Congelamento após `load_stats` permanece explícito quando a camada não estava congelada:

```python
linear_layer.save_stats("cfp-stats.pt")
linear_layer.load_stats("cfp-stats.pt")
linear_layer.freeze_ds_stats()
```

## 4. Compatibilidade corrigida no MPS

Os três casos que eram xfail na P0 agora têm asserções positivas: `load_checkpoint(device="mps")`, `load_state_dict` em camada MPS e `load_stats` nessa camada. Os momentos float64 são desserializados e mantidos em CPU; apenas parâmetros e derivados suportados são enviados ao acelerador. A restauração de um estado congelado deixa as constantes prontas antes do primeiro forward.

O helper também sincroniza o dispositivo de execução da CFP quando a factory constrói inicialmente em CPU e o checkpoint é carregado para MPS. Nesse caso, payloads legados de outro dispositivo são descartados. Scorers personalizados sem parâmetros também foram restaurados em CPU e MPS. A estrutura retornada pelo helper permanece em CPU; a camada restaurada executa no dispositivo solicitado.

As fixtures binárias da P0 e seus hashes não foram alterados. O teste antigo foi atualizado para exigir sucesso das restaurações corrigidas, e o relato histórico da P0 continua disponível.

## 5. Evidência de correção e recursos

A [validação consolidada](validation.json) e o [resultado completo do pytest](validation-tests.xml) registram:

- **51 testes específicos da P1**, incluindo todos os modos estatísticos, CPU/MPS, preparação sem referência ao modelo, liberação de payloads, compartilhamento, snapshots incompatíveis, falha no meio da passagem, sampler/drop_last, múltiplos canais e árvores;
- **64 casos de referência da P0**, com os valores esperados originais e os três casos MPS convertidos em verificações positivas;
- todos os demais testes centrais de Python da biblioteca, incluindo **19 gradchecks**, bindings, fachada morfológica, dados, cache e contratos públicos.

Spies nos testes tornam uma atualização/redução estatística ou recálculo de momentos durante o treino congelado uma falha. A restauração congelada também foi testada com recálculo de momentos proibido no primeiro forward. A checagem de memória usa weakrefs para demonstrar que o objeto preparado e seus atributos deixam de ser retidos antes da próxima preparação, sem depender apenas do RSS do sistema operacional.

O [ensaio final](runs/fit_cpu_3_validated/result.json) usou o mesmo Python, extensão nativa Release e imagens 0, 1000 e 2040 da P0. A extensão manteve o SHA-256 `c84f7e6cb8b9a6677324de94fd9a8372eedb16e3980ed931379b2caaf5a2ff6c`. Nenhum código C++ ou algoritmo do backend foi alterado. O [ambiente](runs/fit_cpu_3_validated/environment.json) contém os hashes dos fontes Python finais, versão das bibliotecas e estado datado da máquina.

| Imagem | Nós | Dados brutos | Preparação |
| --- | ---: | ---: | ---: |
| 0 | 699.690 | 255,13 MiB | 23.064 s |
| 1000 | 627.731 | 234,81 MiB | 20.158 s |
| 2040 | 601.906 | 227,52 MiB | 20.723 s |

Foram produzidos e combinados **189 resumos**: três imagens × 63 atributos. Construir os resumos levou 0.216 s no total, e combiná-los levou 0.023 s. Leitura, conversão, preparação, redução e instalação totalizaram 64.774 s. O número de nós e os bytes dos dados brutos coincidiram exatamente com a P0 para cada imagem.

O [monitor externo](runs/fit_cpu_3_validated/monitor.json) observou **1.111 GiB de pico RSS**, mínimo de **2.331 GiB disponível no sistema** e nenhuma interrupção por limite. Configuração: um processo, uma thread interna, lote 1, sem workers de leitura; observações a cada 100 ms; limites de 4 GiB de RSS, 1 GiB de memória disponível e 600 segundos. Nenhuma imagem preparada foi persistida em disco ou mantida no cache.

Essa é uma medição de ajuste estatístico em três imagens, não do treinamento completo ou de todo o dataset. O RSS inclui runtime e temporários; picos curtos podem escapar da amostragem. Os tempos incluem instrumentação externa e a máquina estava em uso. Portanto, não se afirma um fator de aceleração em relação à P0. Os ensaios preliminares em `runs/fit_cpu_3` e `runs/fit_cpu_3_final` foram preservados, mas a consolidação usa a execução final, cujos hashes coincidem com os fontes entregues.

## 6. Reprodução e limites desta entrega

Reutilizar o build Release da P0 e seu interpretador. O medidor verifica os imports para impedir que uma instalação editável de outro checkout seja usada inadvertidamente.

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py test -- -q \
  mtlearn/tests/python/test_bindings.py mtlearn/tests/python/test_data.py \
  mtlearn/tests/python/test_cfp_preparation.py mtlearn/tests/python/test_cfp_components.py \
  mtlearn/tests/python/test_cfp_deterministic.py mtlearn/tests/python/test_cfp_validation.py \
  mtlearn/tests/python/test_cfp_cache.py mtlearn/tests/python/test_morphology_api.py \
  mtlearn/tests/python/test_gradchecks.py mtlearn/tests/python/test_version.py \
  mtlearn/tests/python/test_cfp_cache_baseline.py

/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p1.py \
  --output build/cfp-cache-p1-repeat
```

O supervisor recusa sobrescrever um diretório de medição existente. Em outra máquina, ajustar interpretador, build, dataset e limites. CUDA não estava disponível; a validação de acelerador desta etapa foi feita em MPS. Testes específicos dos experimentos TIP2026 em andamento não foram incluídos nesta validação central.

**Na P1, `cfp(images)` ainda reconstrói a morfologia a cada consumo direto.** Ajustar estatísticas sem retenção resolve a memória dessa passagem, mas não implementa reutilização em SSD ou RAM limitada durante o treino. `forward_prepared`, consumo compartilhado de preparações, RAM limitada e proteção do ciclo de vida até o backward pertencem à P2; SSD/retomada à P3; processos paralelos à P4. O notebook será migrado na P5. O cache legado completo continua inadequado ao dataset inteiro nos recursos medidos na P0.
