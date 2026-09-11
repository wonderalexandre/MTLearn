# CFP cache — resultado da etapa P3

Data: 2026-09-09. **P3 concluída; P4–P6 pendentes.**

A MTLearn agora persiste preparações brutas em SSD, retoma uma passagem interrompida e reabre os dados em outra sessão sem reconstruir árvores nem depender do modelo que os preparou. Um manifesto SQLite mantém os vínculos entre amostras lógicas e conteúdo; os resumos por atributo permitem obter estatísticas do treino sem ler todos os tensores brutos.

**Validação final: 523 testes aprovados, zero falhas e zero skips.** O ensaio em resolução original cancelou após uma imagem, retomou somente as duas restantes e reabriu o SSD em processos separados para forward/backward de linear e MLP em CPU e MPS. O armazenamento permaneceu com **zero bytes de retenção em RAM** ao final de cada etapa; dados ativos, RSS e memória do MPS foram medidos separadamente.

## 1. Implementação

| Componente | Entrega |
| --- | --- |
| `DiskStore` | Arquivos CPU versionados, cota explícita de disco, camada RAM limitada opcional, mmap e leitura restrita, checksums, publicação atômica e recuperação. |
| Identidade persistente | SHA-256 dos pixels canônicos e configuração de forma, árvore/interpolação/seeds, atributos, dtype, formato e implementação. |
| `CFPPreprocessor.prepare` | Passagem sequencial por dataset finito, cancelamento entre amostras, retomada por manifesto e retorno leve. O padrão continua sem persistência. |
| `PreparationResult` | Estado, contagem, referência ao manifesto e snapshot completo opcional, com fingerprint da composição/contrato estatístico. |
| `get_statistics_contract` / `store.statistics` | Exportam o contrato escalar e recompõem estatísticas a partir dos resumos, sem pesos ou novas árvores. |
| `PreparedDataset` / `collate_prepared` | Leitura por amostra, verificação opcional pela presença da fonte, targets, imagens sem targets, `Subset`, samplers e lotes densos. |

O código está em [storage](../../../mtlearn/python/mtlearn/layers/cfp/storage/), [preparation](../../../mtlearn/python/mtlearn/layers/cfp/preparation/) e [camada CFP](../../../mtlearn/python/mtlearn/layers/cfp/connected_filter_preprocessing_layer.py). O [guia público](../../source/guides/connected-filter-preprocessing.md#local-ssd-preparation-and-resumption) contém os contratos completos e exemplos. As APIs legadas, contagem do cache legado, formatos de checkpoint e nomes dos parâmetros permanecem os anteriores. O backend e a reconstrução matemática não foram alterados.

## 2. Uso e retomada

```python
from mtlearn.layers.cfp import (
    CFPPreprocessor, DiskStore, PreparedDataset, collate_prepared,
)
from torch.utils.data import DataLoader

preprocessor = CFPPreprocessor.from_layer(linear_layer)
with DiskStore("cfp-cache", max_disk_bytes=4 * 1024**3, max_ram_bytes=0) as store:
    result = preprocessor.prepare(
        train_dataset,
        store=store,
        manifest="train-v1",
        source_version="dataset-split-v1",
        preprocessing_version="original-resolution-u8-v1",
        split="train",
        collect_stats=linear_layer.get_statistics_contract(),
    )
    linear_layer.set_stats(result.statistics)
    mlp_layer.set_stats(result.statistics)
    dataset = PreparedDataset(store, "train-v1", source=train_dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=0, collate_fn=collate_prepared)
    for batch, target in loader:
        response = linear_layer(batch)
        # Loss/backward e consumo pelo MLP usam o mesmo lote preparado.
```

A cota do exemplo é ilustrativa: estimar antes de persistir o dataset completo. Ela inclui arquivos publicados e temporários, com overhead de serialização; SQLite/WAL, alocação do filesystem, page cache e memória de trabalho ficam separados. `store.clear()` apaga apenas a retenção RAM. Atingir a cota interrompe a gravação sem apagar entradas válidas; aumentar a cota e repetir a chamada permite retomar.

Passar `cancel=event` com `is_set()` ou um predicado sem argumentos cancela entre amostras. Um resultado cancelado não contém estatísticas parciais. A retomada repete a mesma chamada com os mesmos contratos, source/order e IDs. Ela verifica o conteúdo e descarta a retenção RAM prévia para revalidar os arquivos duráveis; uma cópia válida em RAM não pode esconder corrupção no SSD.

Para uma nova sessão:

```python
with DiskStore("cfp-cache", readonly=True) as store:
    snapshot = store.statistics("train-v1", linear_layer.get_statistics_contract())
    linear_layer.set_stats(snapshot)
    mlp_layer.set_stats(snapshot)
    dataset = PreparedDataset(store, "train-v1", source=train_dataset)
```

O manifesto guarda o contrato morfológico necessário para reabrir os dados sem um modelo. `source` permite verificar os pixels atuais e associar seus targets. Sem `source`, a leitura representa o snapshot preparado, sem targets e sem afirmar que os arquivos de origem continuam iguais. Datasets de imagens sem targets também são suportados. Fontes alteradas falham explicitamente; escolher outro nome/versão de manifesto reaproveita os conteúdos ainda compatíveis.

Preparar validação/teste com `split="evaluation"` e sem `collect_stats`. A biblioteca recusa obter estatísticas de um manifesto de avaliação. Os samplers do treinamento mudam a ordem de consumo do `PreparedDataset`, preservando a composição usada no ajuste. Para incluir multiplicidade na composição, preparar um `Subset` com as ocorrências desejadas e IDs lógicos distintos.

## 3. Integridade e propriedade

Cada entrada contém tensores CPU e tipos primitivos. O writer grava em temporário exclusivo, sincroniza, verifica formato/tensores/checksums, publica por renomeação atômica e registra no SQLite. O checksum registrado corresponde aos bytes verificados antes da publicação. Uma trava do sistema operacional admite um coordenador escritor e é liberada quando seu processo termina; leitores usam conexões próprias.

Uma nova abertura para escrita remove temporários incompletos pertencentes ao cache e valida/registra arquivos completos órfãos. Entradas registradas como válidas são verificadas quando lidas do disco, usando tamanho e SHA-256, `weights_only=True` e `map_location="cpu"`. Checksums embutidos de tensores e resumos permitem verificar órfãos sem registro final. Corrupção é identificada; a reconstrução a partir de uma fonte verificada é permitida com escritor, sem atualizar estatísticas do modelo. Leitores somente leitura falham quando precisam de reparo.

IDs lógicos nunca substituem fingerprints de conteúdo. A transação de cada amostra registra todos os seus canais/árvores e contribuições juntos. Repetir a tarefa não acrescenta outra contribuição; amostras lógicas distintas com pixels iguais compartilham arquivos, mas contam separadamente. A redução percorre contribuições em ordem de posição/canal/árvore, conservando count/sum/sumsq, min/max, dtype e convenções anteriores. O snapshot só é emitido após conclusão da passagem.

`max_ram_bytes=0` é o padrão do SSD. Entradas maiores que esse orçamento são consumidas sem admissão na RAM persistente. Lotes e autograd continuam proprietários dos dados ativos; fechar o store ou retirar suas referências não invalida o backward. `mmap=True` é o padrão de leitura, e pode ser desativado. Os tensores continuam somente leitura por contrato. O orçamento do cache não limita RSS, buffers de normalização, saídas, temporários nativos ou o allocator do acelerador.

## 4. Evidência de correção

A [validação consolidada](validation.json) e o [XML](validation-tests.xml) registram **77 testes novos da P3**, além dos 446 anteriores, incluindo os 19 gradchecks existentes. Cobertura específica:

- 48 comparações com a P0: 12 configurações × CPU/MPS × mmap ligado/desligado, verificando estatísticas, saídas, losses e gradientes;
- retomada sem duplicação, repetição intencional de conteúdo, cancelamento e transações com múltiplos canais/árvores;
- falhas antes/depois da publicação, órfãos, arquivo incompleto, corrupção e cota sem perda de entradas válidas;
- encerramento abrupto real de subprocesso durante escrita e reutilização em outro subprocesso com construção de árvore proibida;
- alteração de pixels/forma/árvore/atributos/dtype/versões, IDs duplicados, isolamento treino/avaliação, `Subset` e sampler;
- reutilização por linear/MLP, fontes sem targets, precisão dupla, shapes diferentes, escritor único e leitores;
- retomada com arquivo corrompido e cópia ainda válida em RAM, e alteração dos bytes entre publicação e registro.

As fixtures binárias P0 mantiveram os hashes. A extensão nativa Release continua com SHA-256 `c84f7e6cb8b9a6677324de94fd9a8372eedb16e3980ed931379b2caaf5a2ff6c`. Os quatro ensaios finais possuem hashes de fontes iguais aos entregues. CUDA não estava disponível; o acelerador testado foi MPS. Testes particulares dos experimentos TIP2026 não foram incluídos na suíte central.

## 5. Ensaio em resolução original

O [medidor externo](../../../scripts/benchmarks/cfp_cache_p3.py) usou IDs 0, 1000 e 2040 de `205_SA_L3D14M3`, imagens de 2.748 × 2.748 pixels, max-tree, 63 atributos e float32. A cota foi 1 GiB de arquivos de tensores, sem retenção RAM; a execução ficou limitada a um processo de preparação e uma thread interna. Os supervisores amostraram RSS agregado a cada 100 ms, com teto de 4 GiB RSS, piso de 1 GiB disponível e timeout de 600 s. Nenhum limite interrompeu o ensaio.

| Processo independente | Árvores preparadas | Tempo do supervisor | Pico RSS amostrado | Mínimo disponível no sistema |
| --- | ---: | ---: | ---: | ---: |
| Preparação e cancelamento após 1 imagem | 1 | 22.533 s | 1.106 GiB | 2.950 GiB |
| Retomada: 2 imagens restantes | 2 | 43.373 s | 1.156 GiB | 3.124 GiB |
| Reabertura CPU: 3 imagens e dois scorers | 0 | 3.965 s | 1.187 GiB | 3.499 GiB |
| Reabertura MPS: 1 imagem e dois scorers | 0 | 2.566 s | 0.338 GiB | 2.707 GiB |

As estatísticas do recorte diagnóstico coincidiram com a P1 (`rtol=1e-10`, `atol=1e-12`). Esse recorte foi definido como treino do ensaio; não é o split completo do notebook. A retomada deixou exatamente **três contribuições lógicas**. O SSD ocupou **752,414,771 bytes (717.56 MiB)** nos arquivos de tensores, mais 106,496 bytes de SQLite/WAL observados na leitura CPU.

| Imagem | Nós | Buffers brutos | Construção/atributos | Carregamento preparado CPU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 699,690 | 255.13 MiB | 20.063 s | 0.357 s |
| 1000 | 627,731 | 234.81 MiB | 19.659 s | 0.324 s |
| 2040 | 601,906 | 227.52 MiB | 20.210 s | 0.327 s |

O tempo de carregamento inclui leitura/verificação da fonte, SHA-256 do arquivo, leitura mapeada e validação estrutural. Os tempos de construção/atributos excluem persistência, que está incluída no tempo total dos processos. O page cache do sistema não foi esvaziado entre processos; estes dados não representam um ensaio de SSD fisicamente frio. O hash integral acrescenta uma passagem de leitura por arquivo carregado. A checagem atual de cota percorre os arquivos presentes; seu custo de metadados cresce com o número de entradas.

A reabertura CPU processou as três imagens com linear e MLP em **3.965 s**, incluindo importação, leitura, estatísticas, forward e backward, com **zero construções e zero novos resumos por atributo**. Loss e norma dos gradientes nas imagens originais coincidiram com a P2 dentro das tolerâncias estabelecidas. A validação completa de tensores/gradientes usa as fixtures P0; o ensaio grande verifica integração e recursos com MSE, sem afirmar qualidade de segmentação ou uma época de treino concluída.

No MPS, os máximos observados ao fim das etapas foram 0.334 GiB em `current_allocated_memory` e 1.067 GiB em `driver_allocated_memory`. São observações pontuais, não picos contínuos, e não devem ser somadas ao RSS como memórias físicas independentes em uma máquina com memória unificada. Picos breves podem escapar da amostragem e a máquina estava em uso.

As [medidas por imagem](measurements.csv), [consolidação](summary.json) e os registros de [cancelamento](runs/final_cancel/result.json), [retomada](runs/final_resume/result.json), [leitura CPU](runs/final_read_cpu/result.json) e [leitura MPS](runs/final_read_mps/result.json) acompanham o relatório. O cache final de diagnóstico está em `build/cfp-cache-p3-validated`, fora dos artefatos de documentação. Os ensaios anteriores em `cancel_after_one`, `cancel_after_one_validated` e `resume_remaining_two` são preliminares e não entram nesta consolidação.

## 6. Reprodução e limites

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py test -- -q \
  mtlearn/tests/python/test_bindings.py mtlearn/tests/python/test_data.py \
  mtlearn/tests/python/test_cfp_preparation.py mtlearn/tests/python/test_cfp_prepared_runtime.py \
  mtlearn/tests/python/test_cfp_disk_cache.py mtlearn/tests/python/test_cfp_components.py \
  mtlearn/tests/python/test_cfp_deterministic.py mtlearn/tests/python/test_cfp_validation.py \
  mtlearn/tests/python/test_cfp_cache.py mtlearn/tests/python/test_morphology_api.py \
  mtlearn/tests/python/test_gradchecks.py mtlearn/tests/python/test_version.py \
  mtlearn/tests/python/test_cfp_cache_baseline.py

/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p3.py --phase first \
  --cache-dir build/cfp-p3-repeat --output build/p3-repeat-first
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p3.py --phase resume \
  --cache-dir build/cfp-p3-repeat --output build/p3-repeat-resume
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p3.py --phase consume \
  --cache-dir build/cfp-p3-repeat --output build/p3-repeat-read
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p3.py --phase consume --device mps --consume-count 1 \
  --cache-dir build/cfp-p3-repeat --output build/p3-repeat-mps
```

Os diretórios de saída e o cache da primeira passagem devem ser novos. A identidade do backend é conservadora: versões/binários diferentes exigem outro diretório; migração de formato não é automática. P3 usa o `CFPPreprocessor` padrão e consumo com `DataLoader(num_workers=0)`. O manifesto é consultado por amostra e não carrega todos os payloads; sua quantidade de metadados cresce com o dataset.

A [próxima etapa](../../cfp-cache-refactoring-plan.md#p4--processos-de-preparação-com-filas-limitadas) é **P4: preparação em processos com concorrência e filas limitadas**. A migração do notebook continua na P5. Não foi preparado o dataset completo de 2.041 imagens nem alterado seu split/hiperparâmetros; o orçamento integral continua dependendo da estimativa e do espaço disponível registrados na P0.
