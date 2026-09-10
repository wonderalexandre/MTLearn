# CFP cache — resultado da etapa P2

Data: 2026-09-09. **P2 concluída; P3–P6 pendentes.**

A CFP agora consome lotes preparados em CPU e permite compartilhar a mesma morfologia entre linear e MLP. A preparação de lotes usa ausência de retenção por padrão; `MemoryStore(max_bytes=...)` habilita explicitamente um cache LRU limitado por bytes. As estatísticas e os parâmetros permanecem em cada modelo.

**Validação final: 446 testes aprovados, zero falhas e zero skips.** O ensaio em três imagens originais usou cache de **256 MiB**, com pico RSS de **1.278 GiB** e dois descartes LRU. O ensaio de uma imagem original em MPS também concluiu forward/backward para ambos os scorers. Nenhum ensaio atualizou estatísticas durante o consumo.

## 1. Implementação entregue

| Componente | Comportamento |
| --- | --- |
| `PreparedBatch` | Agrupa imagem/canal/árvore, protege mapas, valida dimensões e contratos e mantém identidade lógica independente da chave de armazenamento. |
| `CFPPreprocessor.prepare_batch` | Prepara um lote BCHW; calcula identidade dos pixels canônicos somente quando há retenção explícita. |
| `NullStore` | Entrega preparações sem retê-las; pode informar os bytes alcançáveis por consumidores ativos. |
| `MemoryStore` | LRU em bytes de storages únicos, incluindo views, com entradas acima do orçamento entregues sem admissão. |
| `forward_prepared` / `forward` | Consomem a mesma preparação CPU com normalização e transferência temporárias, preservando forma e ordem das saídas. |
| Extensões | `predict`, regularização, inspeção e `CFPContext` funcionam com dados preparados sem reconstruir árvores. |
| Compatibilidade | O adaptador legado usa a mesma conversão para payloads; APIs, contagens, parâmetros e checkpoints foram preservados. |

Os arquivos principais estão em [preparation](../../../mtlearn/python/mtlearn/layers/cfp/preparation/), [storage](../../../mtlearn/python/mtlearn/layers/cfp/storage/), [executor](../../../mtlearn/python/mtlearn/layers/cfp/runtime/forward_executor.py) e [camada CFP](../../../mtlearn/python/mtlearn/layers/cfp/connected_filter_preprocessing_layer.py). O [guia público](../../source/guides/connected-filter-preprocessing.md#prepared-batches-and-bounded-ram) descreve as APIs e os campos das métricas.

## 2. Uso

```python
from mtlearn.layers.cfp import CFPPreprocessor, MemoryStore

snapshot = linear_layer.fit_stats(trainloader)
mlp_layer.set_stats(snapshot)
preprocessor = CFPPreprocessor.from_layer(linear_layer)
store = MemoryStore(max_bytes=256 * 1024**2)

for images, targets in trainloader:
    batch = preprocessor.prepare_batch(images, store=store)
    for layer, optimizer in ((linear_layer, linear_optimizer),
                             (mlp_layer, mlp_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        response = layer.forward_prepared(batch)
        loss = criterion(response, targets.to(response.device))
        loss = loss + layer.regularization_penalty(batch)
        loss.backward()
        optimizer.step()
    del batch
```

Omitir `store` dá preparação temporária, sem hashing para reutilização. Também é possível chamar `layer(batch)`, `layer.predict(batch)` e `layer.inspect_prepared_sample(batch, batch_index=0, channel=0)`. Para estatísticas carregadas por `load_stats`, chamar `freeze_ds_stats` antes do novo consumo preparado. Um lote denso exige mesma forma espacial; usar lote 1 para imagens grandes ou tamanhos variados.

Os dados brutos são somente leitura por contrato. Cada modelo normaliza com seu snapshot e usa seus próprios parâmetros. Uma preparação com atributos adicionais pode atender a um consumidor que usa um subconjunto, desde que árvore e dtype coincidam. A chave RAM considera pixels após a conversão canônica uint8, forma, árvore, atributos, dtype e implementação no processo; IDs lógicos e pesos não invalidam dados fixos.

## 3. Memória e ciclo de vida

O orçamento limita somente a propriedade dos buffers brutos pelo cache. `retained_bytes` conta storages únicos, `active_bytes` os buffers alcançáveis pelos handles retornados aos consumidores, e `live_bytes` a união desses dois conjuntos. `active_not_retained_bytes` identifica dados em uso que já saíram do cache. Os contadores não somam duas vezes memória compartilhada entre views, handles ou entradas.

Descartar uma entrada não altera buffers nem invalida o lote ativo. Os tensores necessários ao gradiente continuam pertencendo ao autograd. Um token salvo no grafo mantém o handle da preparação entre forward e backward; backward comum libera esse handle, enquanto `retain_graph=True` preserva os dados salvos. Dois modelos têm grafos e propriedade independentes sobre a preparação compartilhada.

Essas métricas excluem imagens de entrada, targets, objetos Python, atributos normalizados, saídas, workspace nativo, cópias no acelerador e reservas do allocator. Consumidores personalizados que guardam tensores individuais após soltar o handle ficam fora da contagem de handles ativos. O limite de RAM do cache **não é um limite de RSS**. O tamanho do lote e a quantidade de grafos mantidos pela aplicação continuam determinando a memória de trabalho.

## 4. Validação de correção

A [validação consolidada](validation.json) e o [XML do pytest](validation-tests.xml) registram **94 testes novos da P2**, além dos 352 da P1, incluindo **19 gradchecks** existentes. Os novos testes cobrem:

- 72 comparações com as fixtures binárias imutáveis da P0: 12 configurações × CPU/MPS × temporário/RAM/entrada acima do orçamento;
- saídas, losses e gradientes após remover o cache e a referência do chamador antes do backward;
- compartilhamento entre linear/MLP, troca de estatísticas sem alterar dados brutos e parâmetros independentes;
- múltiplos canais, árvores e specs, constraints, regularizadores, contexto, predição e inspeção;
- LRU com storages compartilhados, backing allocations de views, substituição, orçamento zero e 80 entradas sucessivas;
- identidade por conteúdo, alteração de pixels e resolução, mudança de atributos/dtype/árvore e reordenação dos IDs;
- retenção de grafos, descarte sem backward, inferência sem gradiente, dois consumidores, precisão dupla, exceções e checkpoint sem dependência do armazenamento.

Spies tornam uma reconstrução de árvore ou atualização/redução estatística durante o consumo preparado uma falha. As fixtures P0 mantiveram seus hashes. Não houve alteração em C++, backend morfológico ou reconstrução matemática; a extensão Release mantém SHA-256 `c84f7e6cb8b9a6677324de94fd9a8372eedb16e3980ed931379b2caaf5a2ff6c`. CUDA não estava disponível; o acelerador validado foi MPS. Testes específicos dos experimentos TIP2026 não integram esta validação central.

## 5. Ensaio em resolução original

O [medidor externo](../../../scripts/benchmarks/cfp_cache_p2.py) processou imagens de 2.748 × 2.748 pixels, max-tree, 63 atributos, float32 e lote 1. Cada entrada foi consultada novamente logo após a preparação para verificar um acerto de cache, e consumida sequencialmente por linear e MLP (8 unidades, tanh). Os snapshots foram reconstruídos a partir dos escalares do [ajuste de treino da P1](../p1/runs/fit_cpu_3_validated/result.json), cuja origem e hash foram registrados. Nenhuma árvore foi carregada daquele relatório.

| Dispositivo | Imagem | Nós | Dados brutos | Preparação sem entrada | Consulta com acerto |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 0 | 699,690 | 255.13 MiB | 20.213 s | 4.439 ms |
| CPU | 1000 | 627,731 | 234.81 MiB | 21.070 s | 5.227 ms |
| CPU | 2040 | 601,906 | 227.52 MiB | 21.235 s | 5.264 ms |
| MPS | 0 | 699,690 | 255.13 MiB | 19.304 s | 4.111 ms |

Os números de nós e bytes coincidem com a P0/P1. O limite de 256 MiB admite uma imagem por vez neste recorte: a maior ocupa 255,13 MiB. Em CPU houve três preparações, três acertos e dois descartes LRU. Após o último forward de MLP, o ensaio apagou o cache e a referência do lote; o backward concluiu e os bytes ativos passaram a zero. As duas camadas terminaram com zero amostras no cache legado.

| Medida | CPU, três imagens | MPS, uma imagem |
| --- | ---: | ---: |
| Tempo total observado pelo supervisor | 65.736 s | 21.966 s |
| Pico RSS amostrado | 1.278 GiB | 1.150 GiB |
| Mínimo de memória disponível no sistema | 2.050 GiB | 2.704 GiB |
| Máximo de buffers retidos no cache | 255.13 MiB | 255.13 MiB |
| Atualizações estatísticas no consumo | 0 | 0 |
| Bytes brutos retidos/ativos ao final | 0 / 0 | 0 / 0 |

No MPS, os máximos **observados ao fim das etapas** foram 0.362 GiB em `current_allocated_memory` e 1.067 GiB em `driver_allocated_memory`. Não são picos contínuos e não devem ser somados ao RSS como se fossem domínios independentes de memória unificada.

As [medidas por imagem](measurements.csv), [consolidação](summary.json), [execução CPU](runs/shared_cpu_3/result.json) e [execução MPS](runs/shared_mps_1/result.json) acompanham fontes, ambiente, logs e amostras. Cada supervisor verificou RSS agregado a cada 100 ms, com limites de 4 GiB RSS, 1 GiB disponível e 600 s; nenhum limite interrompeu a execução. Picos mais curtos podem escapar da amostragem, e a máquina estava em uso. Os hashes dos fontes medidos coincidem com os entregues.

A loss MSE deste ensaio serve para verificar forward/backward e recursos; não avalia qualidade de segmentação nem representa uma época do notebook. Tempos de acerto medem validação, hashing e resolução do lote, sem nova árvore; não incluem forward/backward. São observações deste equipamento e recorte, sem promessa de fator de aceleração geral.

## 6. Reprodução e próxima etapa

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py test -- -q \
  mtlearn/tests/python/test_bindings.py mtlearn/tests/python/test_data.py \
  mtlearn/tests/python/test_cfp_preparation.py mtlearn/tests/python/test_cfp_prepared_runtime.py \
  mtlearn/tests/python/test_cfp_components.py mtlearn/tests/python/test_cfp_deterministic.py \
  mtlearn/tests/python/test_cfp_validation.py mtlearn/tests/python/test_cfp_cache.py \
  mtlearn/tests/python/test_morphology_api.py mtlearn/tests/python/test_gradchecks.py \
  mtlearn/tests/python/test_version.py mtlearn/tests/python/test_cfp_cache_baseline.py

/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p2.py \
  --output build/cfp-cache-p2-repeat-cpu
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p2.py --device mps --image-ids 0 \
  --output build/cfp-cache-p2-repeat-mps
```

Os diretórios de medição devem ser novos. A [sequência aprovada](../../cfp-cache-refactoring-plan.md#p3--ssd-identidade-e-retomada) segue com **P3: armazenamento em SSD, identidades persistentes, manifesto e retomada**. Stores RAM são locais a um processo e de acesso sequencial. Para um dataset muito maior que o orçamento, faltas no cache ainda exigem reconstrução; a P2 resolve retenção e compartilhamento, mas não elimina esse custo. Paralelismo permanece na P4 e a migração do notebook na P5. As APIs legadas continuam com seu comportamento explícito de pré-carregamento.
